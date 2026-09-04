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
#include <string>
#include <vector>

namespace {

struct Entry {
    const char* agent;
    int radius;
    int factions;
};

// MUST match tools/compare_engines.py's MATRIX, in the same order - the two
// outputs are compared entry by entry. Only the natively implemented agents
// appear; the rest arrive at M6b.
const Entry kMatrix[] = {
    {"random", 7, 8},   {"random", 4, 4},    {"random", 5, 6},    {"random", 8, 8},
    {"greedy", 7, 8},   {"greedy", 4, 4},    {"greedy", 5, 6},    {"greedy", 8, 8},
    {"heuristic", 7, 8},{"vanguard", 7, 8},  {"marshal", 7, 8},   {"marshal", 5, 6},
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
        {"random", 7, 8}, {"greedy", 7, 8},   {"heuristic", 7, 8},
        {"vanguard", 7, 8}, {"marshal", 7, 8},
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

}  // namespace

int main(int argc, char** argv) {
    const std::string mode = argc > 1 ? argv[1] : "matrix";
    if (mode == "matrix") return run_matrix();
    if (mode == "bench") return run_bench();
    std::fprintf(stderr, "usage: oo_tournament [matrix|bench]\n");
    return 2;
}
