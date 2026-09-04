// Terrain-generation and capital-setup parity.
//
// The terrain half is the M4 gate and the strongest check in the suite so far:
// nothing is replayed, nothing is fed in. A radius and a seed go in, and the C++
// engine must produce the identical map - every hex, plus the generation log in
// the same order. That is only possible if the RNG, the bag weighting order, the
// blob shape rules, the island check and the round structure all agree exactly.
//
// Usage: test_setup <path-to-setup_cases.txt>

#include "oo/placement.hpp"
#include "oo/setup.hpp"
#include "oo/state_io.hpp"

#include <cstdio>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

namespace {

int g_failures = 0;

void fail(const std::string& what) {
    if (++g_failures <= 20) std::cerr << "FAIL: " << what << "\n";
}

// --- setup decision replay --------------------------------------------------

struct Decision {
    char kind = '?';  // L placement, D draft, W swap
    int a = 0, b = 0;
    // The legal set Python offered. Compared, not just used: a mask that is
    // merely LARGER than the correct one still validates the recorded choice, so
    // comparing only choices leaves legal_placement_mask untested. Mutation
    // testing caught exactly that - the edge ban could be dropped entirely with
    // nothing failing.
    std::vector<int> legal;
};

struct Replay {
    const std::vector<Decision>* decisions = nullptr;
    size_t cursor = 0;
    std::string error;

    const Decision* next(char kind, const char* what) {
        if (!error.empty()) return nullptr;
        if (cursor >= decisions->size()) {
            error = std::string("ran out of decisions: C++ asked for ") + what;
            return nullptr;
        }
        const Decision& d = (*decisions)[cursor];
        if (d.kind != kind) {
            std::ostringstream os;
            os << "decision " << cursor << ": C++ asked for " << what << " (" << kind
               << "), Python recorded '" << d.kind << "'";
            error = os.str();
            return nullptr;
        }
        ++cursor;
        return &d;
    }
};

int replay_placement(const oo::GameState& state, int faction, const bool* legal, void* ctx) {
    Replay& r = *static_cast<Replay*>(ctx);
    const size_t index = r.cursor;
    const Decision* d = r.next('L', "a placement");
    if (!d) return -1;
    if (d->a != faction && r.error.empty()) {
        std::ostringstream os;
        os << "decision " << index << ": placement asked for faction " << faction
           << ", Python recorded " << d->a;
        r.error = os.str();
    }
    std::vector<int> got;
    for (int i = 0; i < state.num_hexes; ++i) {
        if (legal[i]) got.push_back(i);
    }
    if (got != d->legal && r.error.empty()) {
        std::ostringstream os;
        os << "decision " << index << ": legal placement mask has " << got.size()
           << " hexes, want " << d->legal.size();
        r.error = os.str();
    }
    return d->b;
}

int replay_draft(const oo::GameState&, int faction, const int16_t* pool, int pool_size,
                 void* ctx) {
    Replay& r = *static_cast<Replay*>(ctx);
    const size_t index = r.cursor;
    const Decision* d = r.next('D', "a draft");
    if (!d) return -1;
    if (d->a != faction && r.error.empty()) {
        std::ostringstream os;
        os << "decision " << index << ": draft asked for faction " << faction
           << ", Python recorded " << d->a;
        r.error = os.str();
    }
    std::vector<int> got(pool, pool + pool_size);
    if (got != d->legal && r.error.empty()) {
        std::ostringstream os;
        os << "decision " << index << ": draft pool has " << got.size() << " hexes, want "
           << d->legal.size();
        r.error = os.str();
    }
    return d->b;
}

bool replay_swap(const oo::GameState&, int faction, int, int, int, void* ctx) {
    Replay& r = *static_cast<Replay*>(ctx);
    const size_t index = r.cursor;
    const Decision* d = r.next('W', "a swap");
    if (!d) return false;
    if (d->a != faction && r.error.empty()) {
        std::ostringstream os;
        os << "decision " << index << ": swap asked for faction " << faction
           << ", Python recorded " << d->a;
        r.error = os.str();
    }
    return d->b != 0;
}

const char* kind_name(int k) {
    switch (k) {
        case 0: return "place";
        case 1: return "draft";
        case 2: return "draft_auto";
        case 3: return "keep";
        default: return "swap";
    }
}

int run_terrain(std::istream& in) {
    std::string tag;
    int total = 0;
    in >> tag >> total;
    if (tag != "TERRAIN_CASES") {
        std::cerr << "expected TERRAIN_CASES, got " << tag << "\n";
        return -1;
    }

    std::vector<int8_t> terrain(oo::MAX_HEXES);
    std::vector<oo::TerrainLogEntry> log;
    long long hexes_checked = 0;

    for (int i = 0; i < total; ++i) {
        int radius = 0;
        long long seed = 0;
        in >> tag >> radius >> seed;

        int n = 0;
        in >> tag >> n;  // TERRAIN
        std::vector<int> want(static_cast<size_t>(n));
        for (int h = 0; h < n; ++h) in >> want[static_cast<size_t>(h)];

        int log_n = 0;
        in >> tag >> log_n;  // LOG
        std::vector<oo::TerrainLogEntry> want_log(static_cast<size_t>(log_n));
        for (int e = 0; e < log_n; ++e) {
            int q, r, s, t, round;
            in >> q >> r >> s >> t >> round;
            want_log[static_cast<size_t>(e)] = oo::TerrainLogEntry{
                static_cast<int8_t>(q), static_cast<int8_t>(r), static_cast<int8_t>(s),
                static_cast<int8_t>(t), round};
        }

        const oo::HexGrid& grid = oo::HexGrid::shared(radius);
        oo::Rng rng(seed);
        log.clear();
        oo::generate_terrain(grid, rng, terrain.data(), &log);

        bool ok = true;
        for (int h = 0; h < n && ok; ++h) {
            if (terrain[static_cast<size_t>(h)] != want[static_cast<size_t>(h)]) {
                std::ostringstream os;
                os << "terrain r" << radius << " seed " << seed << ": hex " << h << " is "
                   << int(terrain[static_cast<size_t>(h)]) << ", want "
                   << want[static_cast<size_t>(h)];
                fail(os.str());
                ok = false;
            }
        }
        hexes_checked += n;

        if (ok && log.size() != want_log.size()) {
            std::ostringstream os;
            os << "terrain r" << radius << " seed " << seed << ": log has " << log.size()
               << " entries, want " << want_log.size();
            fail(os.str());
            ok = false;
        }
        for (size_t e = 0; ok && e < log.size(); ++e) {
            const auto& g = log[e];
            const auto& w = want_log[e];
            if (g.q != w.q || g.r != w.r || g.s != w.s || g.terrain != w.terrain ||
                g.round != w.round) {
                std::ostringstream os;
                os << "terrain r" << radius << " seed " << seed << ": log[" << e << "] = ("
                   << int(g.q) << "," << int(g.r) << "," << int(g.s) << ",t" << int(g.terrain)
                   << ",rd" << g.round << "), want (" << int(w.q) << "," << int(w.r) << ","
                   << int(w.s) << ",t" << int(w.terrain) << ",rd" << w.round << ")";
                fail(os.str());
                ok = false;
            }
        }
    }
    std::printf("  terrain: %d maps, %lld hexes\n", total, hexes_checked);
    return total;
}

int run_setup(std::istream& in) {
    std::string tag;
    int total = 0;
    in >> tag >> total;
    if (tag != "SETUP_CASES") {
        std::cerr << "expected SETUP_CASES, got " << tag << "\n";
        return -1;
    }

    auto before = std::make_unique<oo::GameState>();
    auto expected = std::make_unique<oo::GameState>();
    std::vector<oo::PlacementLogEntry> log;

    for (int i = 0; i < total; ++i) {
        std::string name;
        int radius = 0, num_factions = 0;
        long long game_seed = 0, seed = 0;
        in >> tag >> name;
        in >> tag >> radius >> num_factions >> game_seed;  // PARAMS
        in >> tag >> seed;                                  // SEED

        std::string error;
        if (!oo::read_state(in, *before, error)) {
            std::cerr << "setup case " << name << " (before): " << error << "\n";
            return -1;
        }

        // Build the same starting state from the SEED ALONE and check it matches
        // what Python produced. This is what makes each case a full-pipeline test
        // rather than just a placement replay: create_initial_state (terrain
        // generation plus starting gold/kill-XP) has to compose correctly first.
        {
            auto generated = std::make_unique<oo::GameState>();
            oo::create_initial_state(*generated, radius, num_factions, game_seed);
            std::string diff;
            if (!oo::compare_states(*generated, *before, diff)) {
                fail(name + ": create_initial_state from seed alone: " + diff);
                // Fall through and still run the replay, so one bad map does not
                // hide unrelated placement failures.
            }
        }

        int n_decisions = 0;
        in >> tag >> n_decisions;
        std::vector<Decision> decisions(static_cast<size_t>(n_decisions));
        for (int d = 0; d < n_decisions; ++d) {
            Decision& dec = decisions[static_cast<size_t>(d)];
            std::string kind;
            int n_legal = 0;
            in >> kind >> dec.a >> dec.b >> n_legal;
            dec.kind = kind[0];
            dec.legal.resize(static_cast<size_t>(n_legal));
            for (int k = 0; k < n_legal; ++k) in >> dec.legal[static_cast<size_t>(k)];
        }

        if (!oo::read_state(in, *expected, error)) {
            std::cerr << "setup case " << name << " (after): " << error << "\n";
            return -1;
        }

        int log_n = 0;
        in >> tag >> log_n;
        std::vector<oo::PlacementLogEntry> want_log(static_cast<size_t>(log_n));
        for (int e = 0; e < log_n; ++e) {
            int kind, faction, q, r, s, pf, pq, pr, ps;
            in >> kind >> faction >> q >> r >> s >> pf >> pq >> pr >> ps;
            want_log[static_cast<size_t>(e)] = oo::PlacementLogEntry{
                static_cast<oo::PlacementLogEntry::Kind>(kind), static_cast<int8_t>(faction),
                static_cast<int8_t>(q),  static_cast<int8_t>(r),
                static_cast<int8_t>(s),  static_cast<int8_t>(pf),
                static_cast<int8_t>(pq), static_cast<int8_t>(pr),
                static_cast<int8_t>(ps)};
        }

        Replay replay;
        replay.decisions = &decisions;
        oo::SetupDecisions sd;
        sd.placement = &replay_placement;
        sd.draft = &replay_draft;
        sd.swap = &replay_swap;
        sd.ctx = &replay;

        oo::Rng rng(seed);
        log.clear();
        oo::run_city_setup(*before, sd, rng, &log);

        if (!replay.error.empty()) {
            fail(name + ": " + replay.error);
            continue;
        }
        if (replay.cursor != decisions.size()) {
            std::ostringstream os;
            os << name << ": C++ consumed " << replay.cursor << " decisions, Python made "
               << decisions.size();
            fail(os.str());
            continue;
        }

        std::string diff;
        if (!oo::compare_states(*before, *expected, diff)) {
            fail(name + ": " + diff);
            continue;
        }

        if (log.size() != want_log.size()) {
            std::ostringstream os;
            os << name << ": placement log has " << log.size() << " entries, want "
               << want_log.size();
            fail(os.str());
            continue;
        }
        for (size_t e = 0; e < log.size(); ++e) {
            const auto& g = log[e];
            const auto& w = want_log[e];
            const bool same = g.kind == w.kind && g.faction == w.faction && g.q == w.q &&
                              g.r == w.r && g.s == w.s &&
                              (w.kind != oo::PlacementLogEntry::kSwap ||
                               (g.placer_faction == w.placer_faction && g.placer_q == w.placer_q &&
                                g.placer_r == w.placer_r && g.placer_s == w.placer_s));
            if (!same) {
                std::ostringstream os;
                os << name << ": placement log[" << e << "] = " << kind_name(g.kind) << " f"
                   << int(g.faction) << " (" << int(g.q) << "," << int(g.r) << "," << int(g.s)
                   << "), want " << kind_name(w.kind) << " f" << int(w.faction) << " ("
                   << int(w.q) << "," << int(w.r) << "," << int(w.s) << ")";
                fail(os.str());
                break;
            }
        }

        std::string problem;
        if (!oo::validate_state(*before, problem)) {
            fail(name + ": output state invalid: " + problem);
        }
    }
    std::printf("  setup: %d cases\n", total);
    return total;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "usage: test_setup <setup_cases.txt>\n";
        return 2;
    }
    std::ifstream in(argv[1]);
    if (!in) {
        std::cerr << "cannot open " << argv[1] << "\n";
        return 2;
    }

    const int terrain = run_terrain(in);
    if (terrain < 0) return 2;
    const int setup = run_setup(in);
    if (setup < 0) return 2;

    std::printf("test_setup: %d terrain maps + %d setup cases, %d failures\n", terrain, setup,
                g_failures);
    return g_failures == 0 ? 0 : 1;
}
