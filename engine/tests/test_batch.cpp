// The batched runner must be indistinguishable from the plain one (M9a).
//
// This is the same discipline that made M8b safe: prove the NEW path reproduces
// the old one exactly before anything about behaviour changes. Here the new path
// is thread-per-game with a blocking gate, and the proof is that running the same
// specs through it with a ScriptedPolicy gives byte-identical results to
// run_games - for every batch size and every level of thread oversubscription.
//
// It matters more than it looks. The gate reorders when decisions are ANSWERED,
// interleaves games that used to run start-to-finish on one thread, and passes
// every decision through a request struct that has to carry each of nine kinds
// faithfully. A dropped field or a mixed-up faction index would show up here and
// almost nowhere else.

#include "oo/batch.hpp"
#include "oo/run.hpp"

#include <cstdio>
#include <iostream>
#include <string>
#include <vector>

namespace {

int g_failures = 0;

void check(bool ok, const std::string& what) {
    if (!ok) {
        std::cerr << "FAIL: " << what << "\n";
        ++g_failures;
    }
}

bool same(const oo::GameResult& a, const oo::GameResult& b, int nf) {
    if (a.winner != b.winner || a.turns != b.turns) return false;
    for (int f = 0; f < nf; ++f) {
        if (a.victory_points[f] != b.victory_points[f]) return false;
    }
    return true;
}

}  // namespace

int main() {
    oo::RunConfig cfg;
    cfg.radius = 5;
    cfg.num_factions = 6;
    cfg.max_turns = 40;

    // Mixed seats and mixed costs, for the same reasons test_run gives: a
    // uniform batch would hide an ordering bug, and equal-cost games would keep
    // the gate's arrival order accidentally tidy.
    const oo::AgentKind cheap[] = {oo::AgentKind::kRandom, oo::AgentKind::kGreedy,
                                   oo::AgentKind::kTurtle, oo::AgentKind::kHussar,
                                   oo::AgentKind::kDenier, oo::AgentKind::kVanguard};
    const oo::AgentKind dear[] = {oo::AgentKind::kTactician, oo::AgentKind::kMarshal,
                                  oo::AgentKind::kTactician, oo::AgentKind::kLegion,
                                  oo::AgentKind::kTactician, oo::AgentKind::kWarlord};

    const int num_games = 48;
    std::vector<oo::GameSpec> specs(num_games);
    for (int g = 0; g < num_games; ++g) {
        specs[g].seed = 31000 + g;
        const oo::AgentKind* pool = (g % 2 == 0) ? cheap : dear;
        for (int f = 0; f < cfg.num_factions; ++f) specs[g].seats[f] = pool[(f + g) % 6];
    }

    // The reference: the ordinary single-threaded runner.
    std::vector<oo::GameResult> want(num_games);
    oo::run_games(cfg, specs.data(), want.data(), num_games, 1);

    int played = 0, decided = 0;
    for (const oo::GameResult& r : want) {
        if (r.turns > 0) ++played;
        if (r.winner >= 0) ++decided;
    }
    check(played == num_games, "every reference game advanced");
    check(decided > 0, "at least one reference game had a winner");

    // Threads x batch size. Includes heavy oversubscription (48 games in flight
    // on a 12-thread machine), a batch size larger than the thread count (must
    // clamp, not hang), and batch size 1 (no batching at all).
    const struct {
        int threads, batch;
    } configs[] = {
        {1, 1}, {2, 1}, {4, 4}, {8, 3}, {12, 12}, {16, 8}, {32, 32}, {48, 64}, {48, 1},
    };

    for (const auto& c : configs) {
        std::vector<oo::GameResult> got(num_games);
        for (oo::GameResult& r : got) {
            r.winner = -99;
            r.turns = -99;
        }
        oo::ScriptedPolicy policy(cfg, specs.data(), num_games);
        oo::BatchRunStats stats;
        oo::run_games_batched(cfg, specs.data(), got.data(), num_games, c.threads, c.batch, policy,
                              &stats);

        int mismatches = 0;
        for (int g = 0; g < num_games; ++g) {
            if (!same(want[g], got[g], cfg.num_factions)) ++mismatches;
        }
        if (mismatches) {
            std::cerr << "FAIL: " << c.threads << " threads / batch " << c.batch << ": "
                      << mismatches << " of " << num_games << " games differ\n";
            ++g_failures;
        }
        check(stats.requests > 0, "the gate saw requests");
        check(stats.max_batch <= c.threads, "a batch never exceeds the games in flight");

        std::printf("  %2d threads, batch %2d -> %lld requests in %lld batches "
                    "(mean %.1f, max %d)%s\n",
                    c.threads, c.batch, stats.requests, stats.batches, stats.mean_batch,
                    stats.max_batch, mismatches ? "  MISMATCH" : "");
    }

    // Batching must actually happen when there is room for it, or the whole
    // exercise is pointless and this test would still pass.
    {
        std::vector<oo::GameResult> got(num_games);
        oo::ScriptedPolicy policy(cfg, specs.data(), num_games);
        oo::BatchRunStats stats;
        oo::run_games_batched(cfg, specs.data(), got.data(), num_games, 32, 16, policy, &stats);
        check(stats.max_batch > 1, "requests are genuinely batched, not answered one by one");
        check(stats.mean_batch > 1.5, "the mean batch is meaningfully above 1");
    }

    std::printf("test_batch: %d games x %d configurations, %d failures\n", num_games,
                int(sizeof(configs) / sizeof(configs[0])), g_failures);
    return g_failures == 0 ? 0 : 1;
}
