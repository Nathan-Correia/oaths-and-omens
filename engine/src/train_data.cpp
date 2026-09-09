#include "oo/train_data.hpp"

#include <cstring>
#include <mutex>

namespace oo {

int action_bits(DecisionKind kind, int num_hexes, int num_factions) {
    switch (kind) {
        case DecisionKind::kMovement:
        case DecisionKind::kCavalry:
            // (hex, direction), plus one trailing bit for "pass".
            return num_hexes * NUM_DIRECTIONS + 1;
        case DecisionKind::kBuy:
            return num_hexes * 9;
        case DecisionKind::kPlacement:
        case DecisionKind::kDraft:
            return num_hexes;
        case DecisionKind::kTarget:
            return num_factions;
        case DecisionKind::kResource:
        case DecisionKind::kSwap:
            return 2;
        case DecisionKind::kRectification:
            return MAX_BATTLE_CONTRIB * NUM_UNIT_TYPES;
    }
    return 0;
}

TrainWriter::~TrainWriter() { close(); }

bool TrainWriter::open(const std::string& path, int radius, int num_factions, int num_hexes,
                       std::string& error) {
    position_bytes_ = encoded_bytes(num_hexes);
    record_bytes_ = static_cast<int>(sizeof(SampleHeader)) + position_bytes_ + 2 * kMaskBytes;

    f_ = std::fopen(path.c_str(), "wb");
    if (!f_) {
        error = "cannot open " + path + " for writing";
        return false;
    }
    mutex_ = new std::mutex();

    header_ = TrainFileHeader{};
    header_.magic = kTrainMagic;
    header_.version = kTrainVersion;
    header_.encoding_version = static_cast<uint16_t>(kEncodingVersion);
    header_.radius = static_cast<uint16_t>(radius);
    header_.num_factions = static_cast<uint16_t>(num_factions);
    header_.num_hexes = static_cast<uint16_t>(num_hexes);
    header_.mask_bytes = static_cast<uint16_t>(kMaskBytes);
    header_.record_bytes = static_cast<uint32_t>(record_bytes_);
    header_.position_bytes = static_cast<uint32_t>(position_bytes_);
    header_.num_records = 0;
    std::fwrite(&header_, sizeof(header_), 1, f_);
    return true;
}

void TrainWriter::close() {
    if (!f_) return;
    // The count is not known until the end, so it is patched in place. A reader
    // that trusted the header without this would silently see zero samples.
    header_.num_records = records_;
    std::fseek(f_, 0, SEEK_SET);
    std::fwrite(&header_, sizeof(header_), 1, f_);
    std::fclose(f_);
    f_ = nullptr;
    delete static_cast<std::mutex*>(mutex_);
    mutex_ = nullptr;
}

void TrainWriter::write_game(const std::vector<uint8_t>& samples, int count,
                             const int32_t* victory_points, int num_factions, int winner,
                             int total_turns, int game_id) {
    if (!f_ || count <= 0) return;
    std::lock_guard<std::mutex> lock(*static_cast<std::mutex*>(mutex_));

    // Stamp the outcome into every sample from this game, then write the block in
    // one call - games are dumped from several threads at once, and interleaving
    // partial records would corrupt the file.
    std::vector<uint8_t> block(samples.begin(),
                               samples.begin() + static_cast<size_t>(count) * record_bytes_);
    int32_t max_vp = 0;
    for (int f = 0; f < num_factions; ++f) {
        if (victory_points[f] > max_vp) max_vp = victory_points[f];
    }
    for (int i = 0; i < count; ++i) {
        SampleHeader h;
        std::memcpy(&h, block.data() + static_cast<size_t>(i) * record_bytes_, sizeof(h));
        const int f = h.faction;
        h.final_vp = static_cast<int16_t>(f >= 0 && f < num_factions ? victory_points[f] : 0);
        h.max_vp = static_cast<int16_t>(max_vp);
        h.won = (winner >= 0 && winner == f) ? 1u : 0u;
        h.total_turns = static_cast<int16_t>(total_turns);
        h.game_id = game_id;
        std::memcpy(block.data() + static_cast<size_t>(i) * record_bytes_, &h, sizeof(h));
    }
    std::fwrite(block.data(), 1, block.size(), f_);
    records_ += static_cast<uint64_t>(count);
}

}  // namespace oo
