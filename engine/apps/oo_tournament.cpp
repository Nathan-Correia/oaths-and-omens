// Native tournament driver - the C++ replacement for tournament.py (PLAN.md §1.2).
//
// At M6a it exists for two jobs: emitting results in the same shape as
// tools/compare_engines.py so native games can be diffed against Python's, and
// timing. It grows matchup/free-for-all reporting at M6c.
//
// Usage:
//   oo_tournament matrix           JSON results for the fixed comparison matrix
//   oo_tournament bench            per-agent timings

#include "oo/agent.hpp"
#include "oo/game.hpp"

#include <chrono>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <string>
#include <vector>

namespace {

struct Entry {
    const char* agent;
    int radius;
    int factions;
};

// MUST match tools/dump_agent_games.py's MATRIX, in the same order. All twelve
// agents are native as of M6b.
const Entry kMatrix[] = {
    {"random", 7, 8},    {"random", 4, 4},   {"random", 5, 6},    {"random", 8, 8},
    {"greedy", 7, 8},    {"greedy", 4, 4},   {"greedy", 5, 6},    {"greedy", 8, 8},
    {"heuristic", 7, 8}, {"heuristic", 5, 6},{"heuristic", 4, 4},
    {"vanguard", 7, 8},  {"vanguard", 5, 6},
    {"marshal", 7, 8},   {"marshal", 5, 6},  {"marshal", 8, 8},
    {"turtle", 7, 8},    {"turtle", 5, 6},
    {"denier", 7, 8},    {"denier", 5, 6},
    {"warlord", 7, 8},   {"warlord", 4, 4},
    {"legion", 7, 8},    {"legion", 5, 6},
    {"hussar", 7, 8},    {"hussar", 4, 4},
    {"sentinel", 7, 8},  {"sentinel", 5, 6},
    {"tactician", 7, 8}, {"tactician", 5, 6},{"tactician", 4, 4},
};
constexpr int kSeedsPerEntry = 5;
constexpr int kBaseSeed = 5000;
constexpr int kMaxTurns = 60;

int run_matrix() {
    std::printf("[");
    bool first = true;
    for (const Entry& e : kMatrix) {
        oo::AgentKind kind;
        if (!oo::agent_kind_from_name(e.agent, kind)) {
            std::fprintf(stderr, "unknown agent %s\n", e.agent);
            return 2;
        }
        for (int s = 0; s < kSeedsPerEntry; ++s) {
            const int seed = kBaseSeed + s;
            oo::AgentSet agents;
            oo::build_agents(agents, kind, e.factions, seed);
            const oo::GameResult r =
                oo::play_game(agents, e.radius, e.factions, seed, kMaxTurns);

            if (!first) std::printf(", ");
            first = false;
            std::printf("{\"agent\": \"%s\", \"radius\": %d, \"factions\": %d, \"seed\": %d, ",
                        e.agent, e.radius, e.factions, seed);
            if (r.winner < 0) {
                std::printf("\"winner\": null, ");
            } else {
                std::printf("\"winner\": %d, ", r.winner);
            }
            std::printf("\"turns\": %d, \"vp\": {", r.turns);
            for (int f = 0; f < e.factions; ++f) {
                std::printf("%s\"%d\": %d", f ? ", " : "", f, r.victory_points[f]);
            }
            std::printf("}}");
        }
    }
    std::printf("]\n");
    return 0;
}

int run_bench() {
    const Entry cases[] = {
        {"random", 7, 8},   {"greedy", 7, 8},  {"heuristic", 7, 8}, {"vanguard", 7, 8},
        {"marshal", 7, 8},  {"turtle", 7, 8},  {"denier", 7, 8},    {"warlord", 7, 8},
        {"legion", 7, 8},   {"hussar", 7, 8},  {"sentinel", 7, 8},  {"tactician", 7, 8},
        {"tactician", 5, 6},
    };
    constexpr int kGames = 5;
    std::printf("%-12s %-8s %5s %9s %9s %7s\n", "agent", "board", "games", "total s", "s/game",
                "turns");
    for (const Entry& e : cases) {
        oo::AgentKind kind;
        oo::agent_kind_from_name(e.agent, kind);
        const auto t0 = std::chrono::steady_clock::now();
        int turns = 0;
        for (int g = 0; g < kGames; ++g) {
            oo::AgentSet agents;
            oo::build_agents(agents, kind, e.factions, 7000 + g);
            turns += oo::play_game(agents, e.radius, e.factions, 7000 + g, kMaxTurns).turns;
        }
        const double dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        char board[16];
        std::snprintf(board, sizeof(board), "r%df%d", e.radius, e.factions);
        std::printf("%-12s %-8s %5d %9.3f %9.4f %7.1f\n", e.agent, board, kGames, dt, dt / kGames,
                    static_cast<double>(turns) / kGames);
    }
    return 0;
}

// tournament.py's _rank_of: 1 + how many factions scored strictly higher.
int rank_of(const oo::GameResult& r, int faction, int num_factions) {
    int rank = 1;
    for (int f = 0; f < num_factions; ++f) {
        if (r.victory_points[f] > r.victory_points[faction]) ++rank;
    }
    return rank;
}

// Builds the per-faction agents for one game, taking each seat from a full set of
// its own kind. Building the full set matters: an agent's generator is seeded
// from the faction index, so seat f's agent must come from a set built for f.
void assemble(oo::AgentSet& out, const std::vector<oo::AgentKind>& per_seat, int num_factions,
              int64_t seed) {
    out.num_factions = num_factions;
    std::vector<oo::AgentSet> built(per_seat.size());
    std::vector<bool> made(per_seat.size(), false);
    for (int f = 0; f < num_factions; ++f) {
        for (size_t k = 0; k < per_seat.size(); ++k) {
            if (per_seat[f] != per_seat[k] || made[k]) continue;
            oo::build_agents(built[k], per_seat[k], num_factions, seed);
            made[k] = true;
            break;
        }
    }
    for (int f = 0; f < num_factions; ++f) {
        for (size_t k = 0; k < per_seat.size(); ++k) {
            if (made[k] && per_seat[k] == per_seat[f] && built[k].agents[f]) {
                out.agents[f] = std::move(built[k].agents[f]);
                break;
            }
        }
    }
}

// challenger holds ONE seat, baseline fills the rest. The challenger's seat
// rotates with the game index, cancelling out the fact that placement/draft order
// - and therefore which capital you land, and the settle-order tiebreak - is not
// perfectly symmetric across seats.
int run_matchup(const std::string& challenger, const std::string& baseline, int num_games,
                int radius, int num_factions, int max_turns) {
    oo::AgentKind ck, bk;
    if (!oo::agent_kind_from_name(challenger.c_str(), ck) ||
        !oo::agent_kind_from_name(baseline.c_str(), bk)) {
        std::fprintf(stderr, "unknown agent\n");
        return 2;
    }
    int wins = 0, no_winner = 0;
    long long vp_sum = 0, rank_sum = 0, turns_sum = 0;
    for (int g = 0; g < num_games; ++g) {
        const int seat = g % num_factions;
        const int seed = g;
        std::vector<oo::AgentKind> per_seat(num_factions, bk);
        per_seat[seat] = ck;
        oo::AgentSet agents;
        assemble(agents, per_seat, num_factions, seed);

        const oo::GameResult r = oo::play_game(agents, radius, num_factions, seed, max_turns);
        vp_sum += r.victory_points[seat];
        rank_sum += rank_of(r, seat, num_factions);
        turns_sum += r.turns;
        if (r.winner == seat) ++wins;
        if (r.winner < 0) ++no_winner;
    }
    std::printf("%14s vs %-10s [r=%2d,f=%d] win_rate=%5.1f%% avg_vp=%5.1f avg_rank=%.2f "
                "avg_turns=%5.1f",
                challenger.c_str(), baseline.c_str(), radius, num_factions,
                100.0 * wins / num_games, double(vp_sum) / num_games,
                double(rank_sum) / num_games, double(turns_sum) / num_games);
    if (no_winner) std::printf(" no_winner=%d", no_winner);
    std::printf("\n");
    return 0;
}

// One seat per distinct kind, cycling if there are more seats than kinds, with the
// assignment rotated by game so each kind samples every seat roughly evenly.
int run_ffa(const std::vector<std::string>& keys, int num_games, int radius, int num_factions,
            int max_turns) {
    std::vector<oo::AgentKind> kinds;
    for (const std::string& k : keys) {
        oo::AgentKind kind;
        if (!oo::agent_kind_from_name(k.c_str(), kind)) {
            std::fprintf(stderr, "unknown agent %s\n", k.c_str());
            return 2;
        }
        kinds.push_back(kind);
    }
    const int n_kinds = static_cast<int>(keys.size());
    std::vector<int> seats(n_kinds, 0), wins(n_kinds, 0);
    std::vector<long long> vp(n_kinds, 0), rank(n_kinds, 0);
    long long turns_sum = 0;

    for (int g = 0; g < num_games; ++g) {
        const int rotation = g % n_kinds;
        const int seed = g;
        std::vector<int> which(num_factions);
        std::vector<oo::AgentKind> per_seat(num_factions);
        for (int f = 0; f < num_factions; ++f) {
            which[f] = (f + rotation) % n_kinds;
            per_seat[f] = kinds[which[f]];
        }
        oo::AgentSet agents;
        assemble(agents, per_seat, num_factions, seed);

        const oo::GameResult r = oo::play_game(agents, radius, num_factions, seed, max_turns);
        turns_sum += r.turns;
        for (int f = 0; f < num_factions; ++f) {
            const int k = which[f];
            ++seats[k];
            vp[k] += r.victory_points[f];
            rank[k] += rank_of(r, f, num_factions);
            if (r.winner == f) ++wins[k];
        }
    }
    std::printf("%-12s %6s %9s %8s %9s\n", "agent", "seats", "win_rate", "avg_vp", "avg_rank");
    for (int k = 0; k < n_kinds; ++k) {
        std::printf("%-12s %6d %8.1f%% %8.1f %9.2f\n", keys[k].c_str(), seats[k],
                    seats[k] ? 100.0 * wins[k] / seats[k] : 0.0,
                    seats[k] ? double(vp[k]) / seats[k] : 0.0,
                    seats[k] ? double(rank[k]) / seats[k] : 0.0);
    }
    std::printf("avg_turns=%.1f\n", double(turns_sum) / num_games);
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    const std::string mode = argc > 1 ? argv[1] : "matrix";
    if (mode == "matrix") return run_matrix();
    if (mode == "bench") return run_bench();

    // Defaults mirror tournament.py's DEFAULT_SIZE and MAX_TURNS.
    int radius = 7, num_factions = 8, num_games = 100, max_turns = 200;
    std::vector<std::string> rest;
    for (int i = 2; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() { return (i + 1 < argc) ? argv[++i] : ""; };
        if (a == "--radius") radius = std::atoi(next());
        else if (a == "--factions") num_factions = std::atoi(next());
        else if (a == "--games") num_games = std::atoi(next());
        else if (a == "--max-turns") max_turns = std::atoi(next());
        else rest.push_back(a);
    }

    if (mode == "matchup") {
        if (rest.size() != 2) {
            std::fprintf(stderr, "usage: oo_tournament matchup <challenger> <baseline> [opts]\n");
            return 2;
        }
        return run_matchup(rest[0], rest[1], num_games, radius, num_factions, max_turns);
    }
    if (mode == "ffa") {
        if (rest.empty()) {
            std::fprintf(stderr, "usage: oo_tournament ffa <agent>... [opts]\n");
            return 2;
        }
        return run_ffa(rest, num_games, radius, num_factions, max_turns);
    }
    std::fprintf(stderr, "usage: oo_tournament [matrix|bench|matchup|ffa]\n");
    return 2;
}
