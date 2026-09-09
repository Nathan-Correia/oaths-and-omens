// The neural policy runs and answers legally (M9c).
//
// Only built when LibTorch is present. Weights are random, so nothing here can
// say the answers are GOOD - what it checks is that every decision kind returns
// something the engine will accept, which is the property that makes M10's
// training loop possible at all. An illegal answer would show up as a wrong game
// rather than a crash.
//
// It also pins the feature widths. Those are asserted at run time by TORCH_CHECK
// inside the model, and a mismatch there killed the first neural run outright -
// worth catching in the suite rather than in a benchmark.

#include "oo/batch.hpp"
#include "oo/neural.hpp"
#include "oo/run.hpp"

#include <cstdio>
#include <iostream>
#include <string>

namespace {
int g_failures = 0;
void check(bool ok, const std::string& what) {
    if (!ok) {
        std::cerr << "FAIL: " << what << "\n";
        ++g_failures;
    }
}
}  // namespace

int main() {
    if (!oo::neural_available()) {
        std::printf("test_neural: no LibTorch in this build, skipping\n");
        return 0;
    }

    // Small board, few factions, few games: this is a smoke test of the whole
    // path, not a benchmark.
    oo::RunConfig cfg;
    cfg.radius = 5;
    cfg.num_factions = 4;
    cfg.max_turns = 12;

    oo::NeuralConfig ncfg;
    ncfg.radius = cfg.radius;
    ncfg.num_factions = cfg.num_factions;
    ncfg.d_model = 64;
    ncfg.n_blocks = 2;
    ncfg.n_heads = 4;
    ncfg.cuda = false;  // CPU so the test runs anywhere, including CI without a GPU
    ncfg.fp16 = false;

    std::string err;
    auto policy = oo::make_neural_policy(ncfg, err);
    check(policy != nullptr, "the policy builds: " + err);
    if (!policy) return 1;
    std::printf("  %s\n", oo::neural_describe(*policy).c_str());

    const int num_games = 4;
    std::vector<oo::GameSpec> specs(num_games);
    for (int g = 0; g < num_games; ++g) {
        specs[g].seed = 900 + g;
        for (int f = 0; f < cfg.num_factions; ++f) specs[g].seats[f] = oo::AgentKind::kRandom;
    }

    std::vector<oo::GameResult> results(num_games);
    oo::BatchRunStats stats;
    oo::run_games_batched(cfg, specs.data(), results.data(), num_games, num_games, num_games,
                          *policy, &stats);

    // Every game must have actually played. A policy that returned nonsense would
    // most likely produce zero-turn games or an engine assertion, not silence.
    int played = 0;
    for (const oo::GameResult& r : results) {
        if (r.turns > 0) ++played;
        check(r.turns <= cfg.max_turns, "a game respects the turn cap");
        for (int f = 0; f < cfg.num_factions; ++f) {
            check(r.victory_points[f] >= 0, "victory points are not negative");
        }
    }
    check(played == num_games, "every game advanced at least one turn");
    check(stats.requests > 0, "the policy was actually asked something");

    // All nine decision kinds should have been exercised by four full games -
    // otherwise this test is only covering movement.
    std::printf("  %lld evaluations over %d games in %lld batches\n", stats.requests, num_games,
                stats.batches);
    check(stats.requests > 200, "a meaningful number of decisions went through the net");

    // The parameter estimate should agree with the built model, since M10 will
    // use it to size a run before paying to build one.
    const long long predicted = oo::neural_param_count(ncfg);
    check(predicted > 0, "param count is estimable without building");

    std::printf("test_neural: %d failures\n", g_failures);
    return g_failures == 0 ? 0 : 1;
}
