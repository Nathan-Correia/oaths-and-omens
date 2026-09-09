// Native agents vs the reference Python agents, over whole games.
//
// The strongest available check on M6a: nothing is replayed and no decisions are
// fed in. A (agent, radius, factions, seed) tuple goes in, and the native agents
// must drive the native engine to the identical winner, turn count and per-faction
// victory points that the Python agents got on the Python engine.
//
// Usage: test_agents <path-to-agent_games.txt> [--rewrite <out>]

#include "oo/agent.hpp"
#include "oo/game.hpp"

#include "rewrite.hpp"

#include <cstdio>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

namespace {
constexpr int kMaxTurns = 60;  // must match tools/dump_agent_games.py
}

int main(int argc, char** argv) {
    std::string rewrite_path;
    const bool rewriting = oo_test::take_rewrite_flag(argc, argv, rewrite_path);

    if (argc < 2) {
        std::cerr << "usage: test_agents <agent_games.txt>\n";
        return 2;
    }
    std::ifstream in(argv[1]);
    if (!in) {
        std::cerr << "cannot open " << argv[1] << "\n";
        return 2;
    }

    std::string tag;
    int total = 0;
    in >> tag >> total;
    if (tag != "AGENT_GAMES") {
        std::cerr << "malformed file: expected AGENT_GAMES header\n";
        return 2;
    }

    oo_test::Rewriter rw;
    if (rewriting) {
        if (!rw.open(rewrite_path)) return 2;
        rw.out << "AGENT_GAMES " << total << "\n";
    }

    int passed = 0, failures = 0;
    for (int i = 0; i < total; ++i) {
        std::string agent;
        int radius, factions, seed, want_winner, want_turns;
        in >> agent >> radius >> factions >> seed >> want_winner >> want_turns;
        int want_vp[oo::MAX_FACTIONS] = {};
        for (int f = 0; f < factions; ++f) in >> want_vp[f];

        oo::AgentKind kind;
        if (!oo::agent_kind_from_name(agent.c_str(), kind)) {
            std::cerr << "unknown agent '" << agent << "'\n";
            return 2;
        }

        oo::AgentSet agents;
        oo::build_agents(agents, kind, factions, seed);
        const oo::GameResult r = oo::play_game(agents, radius, factions, seed, kMaxTurns);

        if (rw) {
            // Inputs verbatim, outputs recomputed.
            rw.out << agent << ' ' << radius << ' ' << factions << ' ' << seed << ' ' << r.winner
                   << ' ' << r.turns;
            for (int f = 0; f < factions; ++f) rw.out << ' ' << r.victory_points[f];
            rw.out << "\n";
            ++passed;
            continue;
        }

        std::ostringstream problem;
        if (r.winner != want_winner) {
            problem << "winner " << r.winner << ", want " << want_winner;
        } else if (r.turns != want_turns) {
            problem << "turns " << r.turns << ", want " << want_turns;
        } else {
            for (int f = 0; f < factions; ++f) {
                if (r.victory_points[f] != want_vp[f]) {
                    problem << "vp[" << f << "] " << r.victory_points[f] << ", want " << want_vp[f];
                    break;
                }
            }
        }

        if (problem.str().empty()) {
            ++passed;
        } else if (++failures <= 15) {
            std::cerr << "FAIL " << agent << "-r" << radius << "f" << factions << "s" << seed
                      << ": " << problem.str() << "\n";
        }
    }

    if (rw) {
        std::printf("test_agents: rewrote %d games to %s\n", passed, rewrite_path.c_str());
        return 0;
    }
    std::printf("test_agents: %d/%d games matched the Python agents, %d failures\n", passed, total,
                failures);
    return failures == 0 ? 0 : 1;
}
