// oo_run - the native replacement for run.py (PLAN.md §1.2, §1.3).
//
// Runs games with a board size, faction count, game count and a per-seat agent
// assignment all chosen on the command line. With --games 1 it also writes the
// three replay files, so web_visualizer.html can open the result; with more than
// one game it prints a summary instead (and --replay picks one game to dump).
//
// Examples:
//   oo_run                                        one logged tactician game, r7 f8
//   oo_run --agents tactician,greedy,greedy,random --factions 4
//   oo_run --radius 8 --factions 10 --games 200 --agents tactician,marshal --rotate
//   oo_run --games 50 --agent marshal --replay 7  summary + the replay of game 7
//
// --agents takes one name per seat, and CYCLES if fewer names than factions are
// given, so "--agents tactician,greedy" on 8 factions alternates them.

#include "oo/agent.hpp"
#include "oo/game.hpp"
#include "oo/json.hpp"
#include "oo/log.hpp"
#include "oo/placement.hpp"
#include "oo/run.hpp"
#include "oo/setup.hpp"
#include "oo/turn.hpp"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

namespace {

const char* kAllAgents[] = {"random",  "greedy",  "heuristic", "vanguard",
                            "marshal", "turtle",  "denier",    "warlord",
                            "legion",  "hussar",  "sentinel",  "tactician"};

void usage() {
    std::fprintf(stderr,
                 "usage: oo_run [options]\n"
                 "  --radius N        board radius (default 7; 1-8, see PLAN.md §9)\n"
                 "  --factions N      number of factions (default 8, max 10)\n"
                 "  --games N         games to play (default 1)\n"
                 "  --agent NAME      one agent kind for every seat\n"
                 "  --agents A,B,...  per-seat kinds, cycled if fewer than factions\n"
                 "  --seed N          base seed; game g uses seed+g (default: clock)\n"
                 "  --max-turns N     safety cap, not a rule (default 100)\n"
                 "  --rotate          rotate the assignment by game, to cancel seat bias\n"
                 "  --threads N       worker threads (default: all cores; 1 = no pool)\n"
                 "  --replay K        with --games>1, also write game K's replay files\n"
                 "  --out-dir DIR     where to write the replay files (default .)\n"
                 "  --list-agents     print the available agent kinds and exit\n");
}

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

std::vector<std::string> split_commas(const std::string& s) {
    std::vector<std::string> out;
    size_t start = 0;
    while (start <= s.size()) {
        const size_t comma = s.find(',', start);
        const size_t end = comma == std::string::npos ? s.size() : comma;
        if (end > start) out.push_back(s.substr(start, end - start));
        if (comma == std::string::npos) break;
        start = comma + 1;
    }
    return out;
}

// Writes the three replay files for one game, replaying it with logging on.
bool write_replay(const std::vector<oo::AgentKind>& per_seat, int radius, int num_factions,
                  int64_t seed, int max_turns, const std::string& out_dir) {
    auto state = std::make_unique<oo::GameState>();
    std::vector<oo::TerrainLogEntry> terrain_log;
    oo::create_initial_state(*state, radius, num_factions, seed, &terrain_log);

    oo::AgentSet agents;
    oo::build_mixed_agents(agents, per_seat.data(), num_factions, seed);

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

    const std::string sep =
        out_dir.empty() || out_dir.back() == '/' || out_dir.back() == '\\' ? "" : "/";
    std::string json;
    oo::write_terrain_log_json(json, radius, terrain_log);
    if (!write_file(out_dir + sep + "terrain_gen_log.json", json)) return false;
    oo::write_placement_log_json(json, *state, radius, num_factions, placement_log);
    if (!write_file(out_dir + sep + "city_placement_log.json", json)) return false;
    oo::write_board_state_json(json, *state, radius, num_factions, turns);
    if (!write_file(out_dir + sep + "board_state.json", json)) return false;

    std::printf("wrote board_state.json, terrain_gen_log.json and city_placement_log.json "
                "(%zu turns, seed %lld)\n",
                turns.size(), static_cast<long long>(seed));
    return true;
}

}  // namespace

int main(int argc, char** argv) {
    int radius = 7;
    int num_factions = 8;
    int num_games = 1;
    int max_turns = 100;
    long long base_seed = -1;
    int replay_game = -1;
    int num_threads = 0;  // 0 = one per hardware thread

    bool rotate = false;
    std::string out_dir = ".";
    std::vector<std::string> agent_names{"tactician"};

    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() -> std::string { return (i + 1 < argc) ? argv[++i] : ""; };
        if (a == "--radius") radius = std::atoi(next().c_str());
        else if (a == "--factions") num_factions = std::atoi(next().c_str());
        else if (a == "--games") num_games = std::atoi(next().c_str());
        else if (a == "--max-turns") max_turns = std::atoi(next().c_str());
        else if (a == "--seed") base_seed = std::atoll(next().c_str());
        else if (a == "--replay") replay_game = std::atoi(next().c_str());
        else if (a == "--threads") num_threads = std::atoi(next().c_str());
        else if (a == "--rotate") rotate = true;
        else if (a == "--out-dir") out_dir = next();
        else if (a == "--agent") agent_names = {next()};
        else if (a == "--agents") agent_names = split_commas(next());
        else if (a == "--list-agents") {
            for (const char* n : kAllAgents) std::printf("%s\n", n);
            return 0;
        } else {
            std::fprintf(stderr, "unknown option '%s'\n\n", a.c_str());
            usage();
            return 2;
        }
    }

    if (radius < 1 || radius > 8) {
        std::fprintf(stderr, "--radius must be 1-8 (terrain generation cannot fill a "
                             "radius-9 board; see PLAN.md §9)\n");
        return 2;
    }
    if (num_factions < 1 || num_factions > oo::MAX_FACTIONS) {
        std::fprintf(stderr, "--factions must be 1-%d\n", oo::MAX_FACTIONS);
        return 2;
    }
    if (num_games < 1) {
        std::fprintf(stderr, "--games must be at least 1\n");
        return 2;
    }
    if (agent_names.empty()) {
        std::fprintf(stderr, "--agents needs at least one name\n");
        return 2;
    }
    // Validated up front rather than after the batch has run - an out-of-range
    // --replay should not cost you the games first.
    if (replay_game >= num_games) {
        std::fprintf(stderr, "--replay %d is out of range (games are numbered 0-%d)\n",
                     replay_game, num_games - 1);
        return 2;
    }

    // Cycle the given names across the seats.
    std::vector<oo::AgentKind> per_seat(static_cast<size_t>(num_factions));
    for (int f = 0; f < num_factions; ++f) {
        const std::string& name = agent_names[static_cast<size_t>(f) % agent_names.size()];
        if (!oo::agent_kind_from_name(name.c_str(), per_seat[static_cast<size_t>(f)])) {
            std::fprintf(stderr, "unknown agent '%s' (try --list-agents)\n", name.c_str());
            return 2;
        }
    }

    if (base_seed < 0) {
        base_seed = static_cast<long long>(
                        std::chrono::duration_cast<std::chrono::milliseconds>(
                            std::chrono::system_clock::now().time_since_epoch())
                            .count()) %
                    2147483647LL;
    }

    std::printf("radius %d, %d factions, %d game%s, base seed %lld%s\n", radius, num_factions,
                num_games, num_games == 1 ? "" : "s", base_seed, rotate ? ", rotating seats" : "");
    {
        int threads = num_threads <= 0 ? oo::default_thread_count() : num_threads;
        if (threads > num_games) threads = num_games;
        if (threads > 1) std::printf("%d worker threads\n", threads);
    }
    std::printf("seats: ");
    for (int f = 0; f < num_factions; ++f) {
        std::printf("%s%d=%s", f ? " " : "", f, oo::agent_kind_name(per_seat[static_cast<size_t>(f)]));
    }
    std::printf("\n\n");

    // Per-seat and per-kind tallies. Both are useful: seats show whether position
    // mattered, kinds show which agent actually won.
    std::vector<int> seat_wins(static_cast<size_t>(num_factions), 0);
    std::vector<long long> seat_vp(static_cast<size_t>(num_factions), 0);
    int kind_games[oo::kNumAgentKinds] = {};
    int kind_wins[oo::kNumAgentKinds] = {};
    long long kind_vp[oo::kNumAgentKinds] = {};
    long long kind_rank[oo::kNumAgentKinds] = {};
    long long total_turns = 0;
    int no_winner = 0;

    // Every game's spec is built up front. Rotating shifts the assignment one seat
    // per game, so each kind samples every seat roughly evenly - placement/draft
    // order is not perfectly symmetric across seats, and this cancels that out.
    std::vector<oo::GameSpec> specs(static_cast<size_t>(num_games));
    for (int g = 0; g < num_games; ++g) {
        oo::GameSpec& spec = specs[static_cast<size_t>(g)];
        spec.seed = base_seed + g;
        for (int f = 0; f < num_factions; ++f) {
            const int src = rotate ? (f + g) % num_factions : f;
            spec.seats[f] = per_seat[static_cast<size_t>(src)];
        }
    }

    oo::RunConfig cfg;
    cfg.radius = radius;
    cfg.num_factions = num_factions;
    cfg.max_turns = max_turns;

    std::vector<oo::GameResult> results(static_cast<size_t>(num_games));
    const auto t0 = std::chrono::steady_clock::now();
    oo::run_games(cfg, specs.data(), results.data(), num_games, num_threads);
    const double elapsed =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();

    // Tallied afterwards, on one thread, in game order. Keeping the accumulation
    // out of the workers is what makes the summary independent of thread count -
    // the floating-point sums below would otherwise depend on completion order.
    for (int g = 0; g < num_games; ++g) {
        const oo::GameResult& r = results[static_cast<size_t>(g)];
        const oo::GameSpec& spec = specs[static_cast<size_t>(g)];
        total_turns += r.turns;
        if (r.winner < 0) ++no_winner;
        for (int f = 0; f < num_factions; ++f) {
            const int k = static_cast<int>(spec.seats[f]);
            ++kind_games[k];
            kind_vp[k] += r.victory_points[f];
            int rank = 1;
            for (int o = 0; o < num_factions; ++o) {
                if (r.victory_points[o] > r.victory_points[f]) ++rank;
            }
            kind_rank[k] += rank;
            if (r.winner == f) {
                ++kind_wins[k];
                ++seat_wins[static_cast<size_t>(f)];
            }
            seat_vp[static_cast<size_t>(f)] += r.victory_points[f];
        }
    }

    std::printf("%-12s %7s %9s %8s %9s\n", "agent", "seats", "win_rate", "avg_vp", "avg_rank");
    for (int k = 0; k < oo::kNumAgentKinds; ++k) {
        if (kind_games[k] == 0) continue;
        std::printf("%-12s %7d %8.1f%% %8.1f %9.2f\n",
                    oo::agent_kind_name(static_cast<oo::AgentKind>(k)), kind_games[k],
                    100.0 * kind_wins[k] / kind_games[k],
                    double(kind_vp[k]) / kind_games[k],
                    double(kind_rank[k]) / kind_games[k]);
    }

    if (num_games > 1) {
        std::printf("\n%-6s %9s %8s\n", "seat", "win_rate", "avg_vp");
        for (int f = 0; f < num_factions; ++f) {
            std::printf("%-6d %8.1f%% %8.1f\n", f, 100.0 * seat_wins[static_cast<size_t>(f)] / num_games,
                        double(seat_vp[static_cast<size_t>(f)]) / num_games);
        }
    }

    std::printf("\n%d game%s in %.2fs (%.1f games/s), avg %.1f turns", num_games,
                num_games == 1 ? "" : "s", elapsed, num_games / (elapsed > 0 ? elapsed : 1e-9),
                double(total_turns) / num_games);
    if (no_winner) {
        std::printf(", %d hit the %d-turn cap with no winner", no_winner, max_turns);
    }
    std::printf("\n");

    // Replay files: automatic for a single game, opt-in for a batch.
    const int replay = (num_games == 1) ? 0 : replay_game;
    if (replay >= 0) {
        std::vector<oo::AgentKind> this_game(static_cast<size_t>(num_factions));
        for (int f = 0; f < num_factions; ++f) {
            const int src = rotate ? (f + replay) % num_factions : f;
            this_game[static_cast<size_t>(f)] = per_seat[static_cast<size_t>(src)];
        }
        std::printf("\n");
        if (!write_replay(this_game, radius, num_factions, base_seed + replay, max_turns,
                          out_dir)) {
            return 1;
        }
    }
    return 0;
}
