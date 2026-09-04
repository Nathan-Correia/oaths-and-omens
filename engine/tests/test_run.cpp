// run_games determinism (PLAN.md §7).
//
// The contract is that out[i] depends only on specs[i] and the config - never on
// thread count, never on which worker claimed the game, never on scheduling. That
// is what keeps tournaments reproducible and parity testing meaningful once games
// run in parallel, and it is easy to break later by accident: one piece of shared
// mutable state, one agent that seeds from something ambient, one accumulator
// moved into a worker, and results start depending on timing.
//
// This test needs no golden file. It asserts an internal invariant - every thread
// count agrees with the serial run - so it keeps working as the engine's rules
// change, unlike the reference-data tests.

#include "oo/run.hpp"

#include <cstdio>
#include <vector>

namespace {

int g_failures = 0;

void check(bool ok, const char* what) {
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", what);
        ++g_failures;
    }
}

bool same(const oo::GameResult& a, const oo::GameResult& b, int num_factions) {
    if (a.winner != b.winner || a.turns != b.turns) return false;
    for (int f = 0; f < num_factions; ++f) {
        if (a.victory_points[f] != b.victory_points[f]) return false;
    }
    return true;
}

}  // namespace

int main() {
    oo::RunConfig cfg;
    cfg.radius = 5;
    cfg.num_factions = 6;
    cfg.max_turns = 60;

    // The batch is built to defeat two different ways of passing by accident.
    //
    // Seats VARY BY GAME, so swapping two games' results is observable. With a
    // uniform batch every game would be interchangeable and an ordering bug would
    // compare equal to itself.
    //
    // Costs are deliberately LOPSIDED - cheap games (random/greedy) alternate with
    // expensive ones (tactician in most seats, which runs rollouts and is ~40x
    // slower), so workers finish well out of claim order. Both mutations below
    // were caught with a uniform-cost batch too, so this is hardening rather than
    // a fix for a known gap: it widens the window in which a result written by
    // completion order, or a worker reusing another's buffer, diverges from the
    // serial baseline instead of coincidentally matching it.
    //
    // Verified by mutation, since a determinism test that cannot fail is worthless:
    //   - seeding per thread (spec.seed + a shared tick) -> 146 failures
    //   - writing out[] in completion order rather than by game index -> 731
    // A third attempt, incrementing the completion counter inside the play_one
    // ARGUMENT list, was not caught - correctly, because argument evaluation
    // happens at claim time, so that mutation did not actually reorder anything.
    const oo::AgentKind cheap[] = {oo::AgentKind::kRandom, oo::AgentKind::kGreedy,
                                   oo::AgentKind::kTurtle, oo::AgentKind::kHussar,
                                   oo::AgentKind::kDenier, oo::AgentKind::kVanguard};
    const oo::AgentKind dear[] = {oo::AgentKind::kTactician, oo::AgentKind::kTactician,
                                  oo::AgentKind::kTactician, oo::AgentKind::kMarshal,
                                  oo::AgentKind::kLegion,    oo::AgentKind::kTactician};
    const int num_games = 96;
    std::vector<oo::GameSpec> specs(num_games);
    for (int g = 0; g < num_games; ++g) {
        specs[g].seed = 7000 + g;
        const oo::AgentKind* pool = (g % 2 == 0) ? cheap : dear;
        for (int f = 0; f < cfg.num_factions; ++f) {
            specs[g].seats[f] = pool[(f + g) % 6];
        }
    }

    std::vector<oo::GameResult> serial(num_games);
    oo::run_games(cfg, specs.data(), serial.data(), num_games, 1);

    // A game that ended before turn 1, or with no variety in winners, would make
    // the comparison below vacuous - a broken run_games that wrote nothing at all
    // would "match" itself. Prove the batch actually played.
    int played = 0, decided = 0;
    for (const oo::GameResult& r : serial) {
        if (r.turns > 0) ++played;
        if (r.winner >= 0) ++decided;
    }
    check(played == num_games, "every game advanced at least one turn");
    check(decided > 0, "at least one game reached a winner");

    const int thread_counts[] = {2, 3, 4, 7, 12, 33, 0 /* = hardware */};
    for (int nt : thread_counts) {
        // Prefilled with a value the engine cannot produce, so a game that is
        // never written is caught rather than comparing equal by luck.
        std::vector<oo::GameResult> parallel(num_games);
        for (oo::GameResult& r : parallel) {
            r.winner = -99;
            r.turns = -99;
        }
        oo::run_games(cfg, specs.data(), parallel.data(), num_games, nt);

        for (int g = 0; g < num_games; ++g) {
            if (!same(serial[g], parallel[g], cfg.num_factions)) {
                std::fprintf(stderr,
                             "FAIL: game %d differs at %d threads: "
                             "serial(winner=%d turns=%d) parallel(winner=%d turns=%d)\n",
                             g, nt, serial[g].winner, serial[g].turns, parallel[g].winner,
                             parallel[g].turns);
                ++g_failures;
            }
        }
    }

    // Ordering must follow the SPEC array, not completion order: shuffling the
    // specs must permute the results the same way. This is what catches a worker
    // that writes to a shared cursor's index instead of its game's index.
    std::vector<oo::GameSpec> reversed(specs.rbegin(), specs.rend());
    std::vector<oo::GameResult> rev_out(num_games);
    oo::run_games(cfg, reversed.data(), rev_out.data(), num_games, 0);
    for (int g = 0; g < num_games; ++g) {
        if (!same(serial[num_games - 1 - g], rev_out[g], cfg.num_factions)) {
            std::fprintf(stderr, "FAIL: reversed batch mismatch at %d\n", g);
            ++g_failures;
        }
    }

    std::printf("test_run: %d games x %d thread counts, %d failures\n", num_games,
                static_cast<int>(sizeof(thread_counts) / sizeof(thread_counts[0])) + 1,
                g_failures);
    return g_failures == 0 ? 0 : 1;
}
