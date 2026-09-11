// Training-sample format (PLAN.md §12.2, M10b).
//
// One file per (radius, num_factions) bucket, so every record is the same size
// and the trainer can memory-map it and index arithmetically - no offset table,
// no per-record parsing, and shuffling is just permuting indices.
//
// A record is:
//     SampleHeader           16 bytes
//     EncodedPosition prefix  encoded_bytes(num_hexes)   (§10.4 wire form)
//     legal mask              kMaskBytes
//     chosen mask             kMaskBytes
//
// WHY TWO MASKS RATHER THAN AN ACTION INDEX. Most decisions pick one thing, but
// `decide_buy` picks a SUBSET and `decide_rectification` picks a multiset. A
// single index cannot express those, and giving each kind its own label layout
// would mean nine parsers on the Python side. A bitmask over the kind's action
// space expresses all of them uniformly: one bit for a single choice, several
// for a set. Rectification loses its per-unit COUNTS this way - it becomes
// "which (slot, unit type) pairs were sent back" - which is a real
// approximation and is recorded as such in §12.2.
//
// The value target is stamped after the game ends, so samples are buffered per
// game and written as a block.

#pragma once

#include "oo/config.hpp"
#include "oo/encode.hpp"
#include "oo/policy.hpp"

#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

namespace oo {

// Widest action space over all nine kinds: buy is (hex, one of 9 variants).
inline constexpr int kMaxActionBits = MAX_HEXES * 9;
inline constexpr int kMaskBytes = (kMaxActionBits + 7) / 8;  // 373

inline constexpr uint32_t kTrainMagic = 0x4f4f5431;  // "OOT1"
inline constexpr int kTrainVersion = 2;

#pragma pack(push, 1)
// One file header, then nothing but fixed-size records.
struct TrainFileHeader {
    uint32_t magic;
    uint16_t version;
    uint16_t encoding_version;  // must match kEncodingVersion or the samples are
                                // being read against a different feature layout
    uint16_t radius;
    uint16_t num_factions;
    uint16_t num_hexes;
    uint16_t mask_bytes;
    uint32_t record_bytes;
    uint32_t position_bytes;
    uint64_t num_records;  // patched on close
};

struct SampleHeader {
    uint8_t kind;       // DecisionKind
    uint8_t step;       // movement / cavalry sub-step
    uint8_t won;        // 1 if THIS SAMPLE'S acting faction won
    int8_t faction;     // the acting faction, so the outcome can be per-sample.
                        // The position itself is already relative to it (slot 0
                        // is always "me"), but the value target is not derivable
                        // from the position - it needs the game's result for
                        // this seat specifically, not the winner's.
    int16_t final_vp;   // acting faction's victory points at game end
    int16_t max_vp;     // the winning score in that game, for normalising
    int16_t turn;       // when in the game this decision happened
    int16_t total_turns;
    int32_t game_id;

    // THE VP TRAJECTORY AHEAD OF THIS DECISION (v2).
    //
    // Training on win/loss alone failed, measurably: a terminal result spread
    // over a sixteen-turn game leaves `R - V(s)` dominated by future
    // randomness rather than by the action taken (§12.3). The fix is not a
    // denser reward for its own sake, it is a SHORTER credit horizon - an
    // action that founds an outpost should be paid within a turn or two.
    //
    // Several horizons are stored rather than one precomputed return, because
    // regenerating this corpus costs 32 GB and a discount rate is exactly the
    // kind of thing worth re-tuning without paying that again.
    //
    // Victory points are the literal win condition here (kVpToWin), not a
    // correlate of it, which is why shaping on them carries little of the usual
    // reward-hacking risk.
    int16_t vp_now;   // this faction's VP when the decision was taken
    int16_t vp_p1;    // ...at the end of turn+1, clamped to the last turn played
    int16_t vp_p2;
    int16_t vp_p3;
    int16_t vp_p5;
    int16_t pad2[3];
};
#pragma pack(pop)

static_assert(sizeof(TrainFileHeader) == 32, "file header layout is on-disk format");
static_assert(sizeof(SampleHeader) == 32, "sample header layout is on-disk format");

// Number of valid action bits for a decision kind on a board of `num_hexes`.
// Bits past this are always zero in both masks.
int action_bits(DecisionKind kind, int num_hexes, int num_factions);

// Appends a bit-set to `mask`, which must be kMaskBytes long.
inline void set_bit(uint8_t* mask, int i) { mask[i >> 3] |= static_cast<uint8_t>(1u << (i & 7)); }
inline bool get_bit(const uint8_t* mask, int i) { return (mask[i >> 3] >> (i & 7)) & 1u; }

// Buffers one game's samples, then writes them once the outcome is known.
class TrainWriter {
public:
    TrainWriter() = default;
    ~TrainWriter();

    bool open(const std::string& path, int radius, int num_factions, int num_hexes,
              std::string& error);
    void close();

    // Thread-safe. `samples` is a packed block of `count` records already in
    // final form except for the outcome fields, which are filled in here.
    // `victory_points` is the whole per-faction array at game end and `winner`
    // the winning seat (-1 for none); each sample is stamped with ITS OWN
    // faction's result, read from the header the recorder already wrote.
    void write_game(const std::vector<uint8_t>& samples, int count, const int32_t* victory_points,
                    int num_factions, int winner, int total_turns, int game_id);

    uint64_t records() const { return records_; }
    int record_bytes() const { return record_bytes_; }
    int position_bytes() const { return position_bytes_; }

private:
    std::FILE* f_ = nullptr;
    int record_bytes_ = 0;
    int position_bytes_ = 0;
    uint64_t records_ = 0;
    TrainFileHeader header_{};
    void* mutex_ = nullptr;
};

}  // namespace oo
