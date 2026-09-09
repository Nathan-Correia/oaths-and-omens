// Generates behaviour-cloning data from a scripted teacher (PLAN.md §12.2).
//
//   oo_dump_training --out data/r7f8.bin --games 20000 --agent tactician
//
// Plays whole games with the teacher and records, at every decision point, the
// encoded position, the legal action set, and what the teacher chose. The value
// target is stamped once the game ends.
//
// Games run in parallel and each buffers its own samples, so a game's records
// land contiguously and no two threads interleave partial records. Progress is
// reported by GAMES rather than samples, because that is the number that
// predicts how long the run has left.

#include "oo/agent.hpp"
#include "oo/buy.hpp"
#include "oo/encode.hpp"
#include "oo/movement.hpp"
#include "oo/placement.hpp"
#include "oo/run.hpp"
#include "oo/setup.hpp"
#include "oo/train_data.hpp"
#include "oo/turn.hpp"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <memory>
#include <string>
#include <thread>
#include <vector>

namespace {

void usage() {
    std::fprintf(stderr,
                 "usage: oo_dump_training [options]\n"
                 "  --out PATH        output file (required)\n"
                 "  --games N         games to play (default 1000)\n"
                 "  --agent NAME      teacher (default tactician)\n"
                 "  --radius N        board radius (default 7; §10.3 trains on 5-8)\n"
                 "  --factions N      factions (default 8)\n"
                 "  --seed N          base seed; game g uses seed+g (default 1)\n"
                 "  --max-turns N     safety cap (default 100)\n"
                 "  --every N         record every Nth decision (default 1)\n"
                 "  --threads N       worker threads (default: all cores)\n");
}

// Collects one game's samples. Everything is built in the final on-disk layout
// so writing is a single memcpy per record.
struct Recorder {
    const oo::AgentSet* agents = nullptr;
    const oo::GameState* state = nullptr;
    int record_bytes = 0;
    int position_bytes = 0;
    int num_hexes = 0;
    int num_factions = 0;
    int every = 1;
    long long seen = 0;
    std::vector<uint8_t> buffer;
    int count = 0;

    // Appends a record. `legal` and `chosen` are already-built bitmasks.
    void emit(const oo::DecisionRequest& req, const uint8_t* legal, const uint8_t* chosen) {
        if (++seen % every != 0) return;
        buffer.resize(static_cast<size_t>(count + 1) * record_bytes);
        uint8_t* p = buffer.data() + static_cast<size_t>(count) * record_bytes;

        oo::SampleHeader h{};
        h.kind = static_cast<uint8_t>(req.kind);
        h.step = static_cast<uint8_t>(req.step);
        h.faction = static_cast<int8_t>(req.faction);
        h.turn = static_cast<int16_t>(state->turn_number);
        std::memcpy(p, &h, sizeof(h));
        p += sizeof(h);

        // The position is encoded into a full EncodedPosition and only its
        // prefix stored - the hexes array is the tail, so the prefix is exactly
        // the header, the factions and this board's hexes.
        oo::EncodedPosition enc;
        oo::encode_position(*state, req, enc);
        std::memcpy(p, &enc, static_cast<size_t>(position_bytes));
        p += position_bytes;

        std::memcpy(p, legal, oo::kMaskBytes);
        p += oo::kMaskBytes;
        std::memcpy(p, chosen, oo::kMaskBytes);
        ++count;
    }
};

struct Masks {
    uint8_t legal[oo::kMaskBytes];
    uint8_t chosen[oo::kMaskBytes];
    Masks() {
        std::memset(legal, 0, sizeof(legal));
        std::memset(chosen, 0, sizeof(chosen));
    }
};

oo::DecisionRequest make_req(const Recorder& r, oo::DecisionKind kind, int faction) {
    oo::DecisionRequest q;
    q.state = r.state;
    q.kind = kind;
    q.faction = faction;
    return q;
}

// --- the recording decision callbacks ---------------------------------------

void rec_buy(const oo::GameState& s, int faction, const oo::LegalBuyActions& legal,
             oo::ChosenBuyActions& out, void* ctx) {
    Recorder& r = *static_cast<Recorder*>(ctx);
    r.agents->get(faction)->decide_buy(s, faction, legal, out);

    Masks m;
    auto variant = [](const oo::BuyAction& a) {
        switch (a.type) {
            case oo::BuyType::kBuyInfantry: return 0;
            case oo::BuyType::kConvertToSpecial: return 1 + int(a.unit_type);
            case oo::BuyType::kBuildOutpost: return 3 + int(a.unit_type);
            default: return 6 + int(a.upgrade);
        }
    };
    for (int i = 0; i < legal.size(); ++i) {
        oo::set_bit(m.legal, legal[i].hex * 9 + variant(legal[i]));
    }
    for (int i = 0; i < out.size(); ++i) {
        oo::set_bit(m.chosen, out[i].hex * 9 + variant(out[i]));
    }
    oo::DecisionRequest q = make_req(r, oo::DecisionKind::kBuy, faction);
    r.emit(q, m.legal, m.chosen);
}

bool rec_move(oo::DecisionKind kind, const oo::GameState& s, int faction, int step,
              const oo::LegalMask& legal, oo::Move& out, void* ctx) {
    Recorder& r = *static_cast<Recorder*>(ctx);
    const bool moved = kind == oo::DecisionKind::kMovement
                           ? r.agents->get(faction)->decide_movement(s, faction, step, legal, out)
                           : r.agents->get(faction)->decide_cavalry(s, faction, step, legal, out);
    Masks m;
    for (int h = 0; h < r.num_hexes; ++h) {
        for (int d = 0; d < oo::NUM_DIRECTIONS; ++d) {
            if (legal.cell[h][d]) oo::set_bit(m.legal, h * oo::NUM_DIRECTIONS + d);
        }
    }
    // The pass action is always available and lives in the trailing bit.
    oo::set_bit(m.legal, r.num_hexes * oo::NUM_DIRECTIONS);
    if (moved) {
        oo::set_bit(m.chosen, out.hex * oo::NUM_DIRECTIONS + out.dir);
    } else {
        oo::set_bit(m.chosen, r.num_hexes * oo::NUM_DIRECTIONS);
    }
    oo::DecisionRequest q = make_req(r, kind, faction);
    q.step = step;
    r.emit(q, m.legal, m.chosen);
    return moved;
}

bool rec_movement(const oo::GameState& s, int faction, int step, const oo::LegalMask& legal,
                  oo::Move& out, void* ctx) {
    return rec_move(oo::DecisionKind::kMovement, s, faction, step, legal, out, ctx);
}
bool rec_cavalry(const oo::GameState& s, int faction, int step, const oo::LegalMask& legal,
                 oo::Move& out, void* ctx) {
    return rec_move(oo::DecisionKind::kCavalry, s, faction, step, legal, out, ctx);
}

int rec_target(const oo::GameState& s, int hex_index, int faction, void* ctx) {
    Recorder& r = *static_cast<Recorder*>(ctx);
    const int t = r.agents->get(faction)->decide_target(s, hex_index, faction);
    Masks m;
    const oo::Battle* b = s.battle_at(hex_index);
    if (b) {
        for (int i = 0; i < b->nslots; ++i) {
            const int f = b->slots[i].faction;
            if (f == oo::NO_FACTION || f == faction) continue;
            oo::set_bit(m.legal, (f - faction + r.num_factions) % r.num_factions);
        }
    }
    if (t >= 0) oo::set_bit(m.chosen, (t - faction + r.num_factions) % r.num_factions);
    oo::DecisionRequest q = make_req(r, oo::DecisionKind::kTarget, faction);
    q.hex = hex_index;
    r.emit(q, m.legal, m.chosen);
    return t;
}

void rec_rectification(const oo::GameState& s, int hex_index, int winner, int cap,
                       oo::SendBack& out, void* ctx) {
    Recorder& r = *static_cast<Recorder*>(ctx);
    r.agents->get(winner)->decide_rectification(s, hex_index, winner, cap, out);
    Masks m;
    const oo::Battle* b = s.battle_at(hex_index);
    if (b) {
        for (int i = 0; i < b->nslots; ++i) {
            if (b->slots[i].faction != winner) continue;
            for (int t = 0; t < oo::NUM_UNIT_TYPES; ++t) {
                if (b->slots[i].units[t] > 0) oo::set_bit(m.legal, i * oo::NUM_UNIT_TYPES + t);
            }
        }
        // Which (slot, unit type) pairs were sent back. Counts are lost here;
        // see the note in train_data.hpp.
        for (int e = 0; e < out.size(); ++e) {
            for (int i = 0; i < b->nslots; ++i) {
                if (b->slots[i].faction != winner) continue;
                if (b->slots[i].origin != out[e].origin_hex) continue;
                for (int t = 0; t < oo::NUM_UNIT_TYPES; ++t) {
                    if (out[e].units[t] > 0) oo::set_bit(m.chosen, i * oo::NUM_UNIT_TYPES + t);
                }
                break;
            }
        }
    }
    oo::DecisionRequest q = make_req(r, oo::DecisionKind::kRectification, winner);
    q.hex = hex_index;
    q.winner = winner;
    q.cap = cap;
    r.emit(q, m.legal, m.chosen);
}

oo::Resource rec_resource(const oo::GameState& s, int faction, int hex_index, void* ctx) {
    Recorder& r = *static_cast<Recorder*>(ctx);
    const oo::Resource res = r.agents->get(faction)->decide_resource_choice(s, faction, hex_index);
    Masks m;
    oo::set_bit(m.legal, 0);
    oo::set_bit(m.legal, 1);
    oo::set_bit(m.chosen, res == oo::kIron ? 1 : 0);
    oo::DecisionRequest q = make_req(r, oo::DecisionKind::kResource, faction);
    q.hex = hex_index;
    r.emit(q, m.legal, m.chosen);
    return res;
}

int rec_placement(const oo::GameState& s, int faction, const bool* legal, void* ctx) {
    Recorder& r = *static_cast<Recorder*>(ctx);
    const int h = r.agents->get(faction)->decide_placement(s, faction, legal);
    Masks m;
    for (int i = 0; i < r.num_hexes; ++i) {
        if (legal[i]) oo::set_bit(m.legal, i);
    }
    if (h >= 0) oo::set_bit(m.chosen, h);
    oo::DecisionRequest q = make_req(r, oo::DecisionKind::kPlacement, faction);
    q.legal_hexes = legal;
    r.emit(q, m.legal, m.chosen);
    return h;
}

int rec_draft(const oo::GameState& s, int faction, const int16_t* pool, int pool_size, void* ctx) {
    Recorder& r = *static_cast<Recorder*>(ctx);
    const int h = r.agents->get(faction)->decide_draft(s, faction, pool, pool_size);
    Masks m;
    for (int i = 0; i < pool_size; ++i) oo::set_bit(m.legal, pool[i]);
    if (h >= 0) oo::set_bit(m.chosen, h);
    oo::DecisionRequest q = make_req(r, oo::DecisionKind::kDraft, faction);
    q.pool = pool;
    q.pool_size = pool_size;
    r.emit(q, m.legal, m.chosen);
    return h;
}

bool rec_swap(const oo::GameState& s, int faction, int leftover, int placer, int placer_hex,
              void* ctx) {
    Recorder& r = *static_cast<Recorder*>(ctx);
    const bool yes = r.agents->get(faction)->decide_swap(s, faction, leftover, placer, placer_hex);
    Masks m;
    oo::set_bit(m.legal, 0);
    oo::set_bit(m.legal, 1);
    oo::set_bit(m.chosen, yes ? 1 : 0);
    oo::DecisionRequest q = make_req(r, oo::DecisionKind::kSwap, faction);
    q.leftover_hex = leftover;
    q.placer_hex = placer_hex;
    q.placer_faction = placer;
    r.emit(q, m.legal, m.chosen);
    return yes;
}

}  // namespace

int main(int argc, char** argv) {
    std::string out_path, agent_name = "tactician";
    int num_games = 1000, radius = 7, num_factions = 8, max_turns = 100, every = 1, threads = 0;
    long long base_seed = 1;

    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() -> std::string { return (i + 1 < argc) ? argv[++i] : ""; };
        if (a == "--out") out_path = next();
        else if (a == "--games") num_games = std::atoi(next().c_str());
        else if (a == "--agent") agent_name = next();
        else if (a == "--radius") radius = std::atoi(next().c_str());
        else if (a == "--factions") num_factions = std::atoi(next().c_str());
        else if (a == "--seed") base_seed = std::atoll(next().c_str());
        else if (a == "--max-turns") max_turns = std::atoi(next().c_str());
        else if (a == "--every") every = std::atoi(next().c_str());
        else if (a == "--threads") threads = std::atoi(next().c_str());
        else {
            std::fprintf(stderr, "unknown option '%s'\n\n", a.c_str());
            usage();
            return 2;
        }
    }
    if (out_path.empty()) {
        usage();
        return 2;
    }
    oo::AgentKind kind;
    if (!oo::agent_kind_from_name(agent_name.c_str(), kind)) {
        std::fprintf(stderr, "unknown agent '%s'\n", agent_name.c_str());
        return 2;
    }
    if (radius < 1 || radius > 8 || num_factions < 1 || num_factions > oo::MAX_FACTIONS ||
        num_games < 1 || every < 1) {
        std::fprintf(stderr, "bad arguments\n");
        return 2;
    }
    if (threads <= 0) threads = oo::default_thread_count();

    const oo::HexGrid& grid = oo::HexGrid::shared(radius);
    oo::TrainWriter writer;
    std::string err;
    if (!writer.open(out_path, radius, num_factions, grid.num_hexes(), err)) {
        std::fprintf(stderr, "%s\n", err.c_str());
        return 1;
    }

    std::printf("teacher %s, radius %d, %d factions, %d games on %d threads\n", agent_name.c_str(),
                radius, num_factions, num_games, threads);
    std::printf("record %d bytes (position %d + masks 2x%d + header %d)\n", writer.record_bytes(),
                writer.position_bytes(), oo::kMaskBytes, int(sizeof(oo::SampleHeader)));
    if (every > 1) std::printf("recording every %d-th decision\n", every);
    std::fflush(stdout);

    std::atomic<int> next_game{0};
    std::atomic<int> done{0};
    const auto t0 = std::chrono::steady_clock::now();

    // A dedicated reporter thread, so progress keeps moving even while a slow
    // tactician game is mid-flight and nothing has been written for a while.
    std::atomic<bool> running{true};
    std::thread reporter([&] {
        while (running.load()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(250));
            const int d = done.load();
            const double el =
                std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            const double frac = double(d) / double(num_games);
            const int width = 34;
            const int filled = int(frac * width);
            std::string bar(static_cast<size_t>(filled), '=');
            bar.resize(width, ' ');
            const double rate = el > 0 ? d / el : 0.0;
            const double eta = rate > 0 ? (num_games - d) / rate : 0.0;
            std::printf("\r[%s] %d/%d games  %.1f g/s  %llu samples  eta %.0fs   ", bar.c_str(), d,
                        num_games, rate, static_cast<unsigned long long>(writer.records()), eta);
            std::fflush(stdout);
        }
    });

    std::vector<std::thread> workers;
    for (int t = 0; t < threads; ++t) {
        workers.emplace_back([&] {
            for (;;) {
                const int g = next_game.fetch_add(1);
                if (g >= num_games) break;
                const int64_t seed = base_seed + g;

                auto state = std::make_unique<oo::GameState>();
                oo::create_initial_state(*state, radius, num_factions, seed);

                oo::AgentSet agents;
                oo::build_agents(agents, kind, num_factions, seed);

                Recorder rec;
                rec.agents = &agents;
                rec.state = state.get();
                rec.record_bytes = writer.record_bytes();
                rec.position_bytes = writer.position_bytes();
                rec.num_hexes = state->num_hexes;
                rec.num_factions = num_factions;
                rec.every = every;

                oo::SetupDecisions sd;
                sd.placement = &rec_placement;
                sd.draft = &rec_draft;
                sd.swap = &rec_swap;
                sd.ctx = &rec;

                oo::TurnDecisions td;
                td.buy = &rec_buy;
                td.movement = &rec_movement;
                td.cavalry = &rec_cavalry;
                td.target = &rec_target;
                td.rectification = &rec_rectification;
                td.resource_choice = &rec_resource;
                td.ctx = &rec;

                oo::Rng rng(seed);
                oo::run_city_setup(*state, sd, rng);
                int turns = 0;
                while (!oo::check_game_end(*state, max_turns)) {
                    oo::run_turn(*state, td, rng);
                    ++turns;
                }

                // Each sample is stamped with ITS OWN acting faction's result,
                // not the game's winner - a value head trained on "did anyone
                // win" would learn nothing about the seat it is playing.
                const int winner = oo::get_game_winner(*state);
                writer.write_game(rec.buffer, rec.count, state->victory_points, num_factions,
                                  winner, turns, g);
                done.fetch_add(1);
            }
        });
    }
    for (std::thread& w : workers) w.join();
    running.store(false);
    reporter.join();

    const double el = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    const uint64_t n = writer.records();
    writer.close();
    std::printf("\nwrote %llu samples to %s (%.2f GB) in %.1fs\n",
                static_cast<unsigned long long>(n), out_path.c_str(),
                double(n) * writer.record_bytes() / 1e9, el);
    return 0;
}
