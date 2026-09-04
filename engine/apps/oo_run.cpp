// oo_run - the native replacement for run.py (PLAN.md §1.2, §1.3).
//
// Plays one logged game and writes the three replay files. This is the piece
// that makes replay Python-free: board_state.json comes out of C++, so
// web_visualizer.html keeps working after everything else is deleted.
//
// Usage:
//   oo_run [--radius N] [--factions N] [--seed N] [--max-turns N]
//          [--agent NAME] [--out-dir DIR]
//
// With no --seed, one is taken from the clock and printed, so a run can be
// reproduced later by passing it back.

#include "oo/agent.hpp"
#include "oo/json.hpp"
#include "oo/log.hpp"
#include "oo/placement.hpp"
#include "oo/setup.hpp"
#include "oo/turn.hpp"

#include <chrono>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

namespace {

bool write_file(const std::string& path, const std::string& contents) {
    // Binary mode: text mode on Windows would translate '\n' to "\r\n" and the
    // output would no longer be byte-identical to Python's.
    std::ofstream f(path, std::ios::binary);
    if (!f) {
        std::fprintf(stderr, "cannot write %s\n", path.c_str());
        return false;
    }
    f.write(contents.data(), static_cast<std::streamsize>(contents.size()));
    return static_cast<bool>(f);
}

}  // namespace

int main(int argc, char** argv) {
    int radius = 7;
    int num_factions = 8;
    int max_turns = 100;
    long long seed = -1;
    std::string agent_name = "tactician";
    std::string out_dir = ".";

    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() { return (i + 1 < argc) ? argv[++i] : ""; };
        if (a == "--radius") radius = std::atoi(next());
        else if (a == "--factions") num_factions = std::atoi(next());
        else if (a == "--seed") seed = std::atoll(next());
        else if (a == "--max-turns") max_turns = std::atoi(next());
        else if (a == "--agent") agent_name = next();
        else if (a == "--out-dir") out_dir = next();
        else {
            std::fprintf(stderr, "unknown option %s\n", a.c_str());
            return 2;
        }
    }

    if (seed < 0) {
        // Same shape as run.py's clock seed, kept inside int32 so it stays a
        // reproducible value to type back in.
        seed = static_cast<long long>(
                   std::chrono::duration_cast<std::chrono::milliseconds>(
                       std::chrono::system_clock::now().time_since_epoch())
                       .count()) %
               2147483647LL;
    }

    oo::AgentKind kind;
    if (!oo::agent_kind_from_name(agent_name.c_str(), kind)) {
        std::fprintf(stderr, "unknown agent '%s'\n", agent_name.c_str());
        return 2;
    }

    auto state = std::make_unique<oo::GameState>();
    std::vector<oo::TerrainLogEntry> terrain_log;
    oo::create_initial_state(*state, radius, num_factions, seed, &terrain_log);

    oo::AgentSet agents;
    oo::build_agents(agents, kind, num_factions, seed);

    oo::Rng rng(seed);
    std::vector<oo::PlacementLogEntry> placement_log;
    oo::SetupDecisions sd = oo::make_setup_decisions(agents);
    oo::run_city_setup(*state, sd, rng, &placement_log);

    oo::TurnDecisions td = oo::make_turn_decisions(agents);
    std::vector<oo::TurnRecord> turns;
    while (!oo::check_game_end(*state, max_turns)) {
        oo::TurnRecord record;
        oo::run_turn_and_log(*state, td, rng, record);
        turns.push_back(std::move(record));
    }

    const std::string sep = out_dir.empty() || out_dir.back() == '/' || out_dir.back() == '\\'
                                ? ""
                                : "/";
    std::string json;

    oo::write_terrain_log_json(json, radius, terrain_log);
    if (!write_file(out_dir + sep + "terrain_gen_log.json", json)) return 1;

    oo::write_placement_log_json(json, *state, radius, num_factions, placement_log);
    if (!write_file(out_dir + sep + "city_placement_log.json", json)) return 1;

    oo::write_board_state_json(json, *state, radius, num_factions, turns);
    if (!write_file(out_dir + sep + "board_state.json", json)) return 1;

    std::printf("Ran %zu turns (seed=%lld, agent=%s), wrote board_state.json, "
                "terrain_gen_log.json, and city_placement_log.json\n",
                turns.size(), seed, agent_name.c_str());
    return 0;
}
