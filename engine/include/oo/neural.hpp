// The neural policy (PLAN.md §10, M9c).
//
// Deliberately free of any torch type, so `oo_run` and anything else can include
// this whether or not LibTorch is present. Everything torch lives in
// src/neural.cpp, which is only compiled when OO_LIBTORCH_DIR is set.
//
// ONE POLICY SERVES ONE BUCKET. An instance is built for a specific
// (radius, num_factions) pair and every position in a batch has the same token
// count, so there is no padding and no attention mask - §10.8's bucketing,
// taken to its conclusion. Serving mixed radii from one process means one
// instance per radius, which is what §12's training loop will want anyway.

#pragma once

#include "oo/policy.hpp"

#include <memory>
#include <string>

namespace oo {

struct NeuralConfig {
    // A TorchScript module exported by train/export.py. REQUIRED - the network
    // is defined in train/model.py and nowhere else, so the engine has no
    // architecture to fall back on. For an untrained one:
    //   python train/export.py --init --out checkpoints/init.ts --radius 7 --factions 8
    std::string weights;

    // The bucket this instance serves. Checked against the checkpoint's own
    // stamped values, which is the difference between a clear refusal and a net
    // that plays like noise.
    int radius = 7;
    int num_factions = 8;

    bool cuda = true;
    bool fp16 = true;

    // Softmax temperature for the single-choice heads. 0 means argmax, which is
    // how the policy PLAYS - deterministic, and what the evaluation harness
    // measures.
    //
    // Self-play needs it above zero. A deterministic policy playing itself only
    // ever visits the actions it already takes, so the games carry no evidence
    // about anything else and there is nothing to learn from them (PLAN.md
    // 12.3). Sampling is what makes a self-play corpus informative.
    float temperature = 0.0f;

    // Seeds the sampling stream. Sampling is deliberately NOT reproducible per
    // game - it is per worker thread - because generating self-play data does
    // not need bit-reproducibility, and making it per game would mean a
    // generator lookup on every decision.
    int64_t sample_seed = 1;

    // The logit above which a buy cell is purchased.
    //
    // The buy head is per-cell BCE, so the decoder takes every cell the network
    // thinks is more likely than not - the mode of a product of independent
    // marginals. That is not a coherent purchase: measured against tactician it
    // buys 4.71 cells where tactician buys 3.96, with recall 93.8% against
    // precision 78.7%, i.e. nearly everything tactician would buy PLUS extras.
    //
    // Exact-set agreement cannot see this. Tactician's buy is stochastic
    // (greedy_buy_common shuffles hexes and picks among equal-scoring infantry),
    // and two draws from tactician agree with each other only 45.2% of the time
    // - exactly what the network already scores, so that metric is pinned at
    // its ceiling and is blind to over-buying. Win rate is not.
    float buy_threshold = 0.0f;

    // ONE-PLY VALUE LOOKAHEAD ON THE BUY DECISION (PLAN.md §12.3).
    //
    // Three self-play objectives in a row produced a flat ladder, and the
    // measurement that explains it is that the buy policy changed on ~32 % of
    // positions while strength did not move. Advantage-weighted regression on
    // on-policy data is self-imitation plus a re-weighting: it cannot propose an
    // action the policy did not already take, so it is not a policy improvement
    // operator, and a self-play loop built on one has nothing to climb.
    //
    // This is the smallest operator that is one. Enumerate k candidate purchase
    // sets, apply each to a copy of the state, score the successor with the
    // value head, keep the best. The result is stronger than the network that
    // proposed it, which is the property cloning needs.
    //
    // Buy specifically, for two reasons. It is the entire measured strength gap
    // (routing buy to the teacher took the policy from 18 % to 69 % against
    // greedy, where movement and cavalry moved it to 17.3 % and 16.0 %). And it
    // fires once per faction per TURN rather than once per unit, so the cost is
    // k extra encodes and one extra batched forward per turn - which is the
    // objection that ruled out MCTS, and it does not apply at this frequency.
    //
    // 0 or 1 disables it. Candidate 0 is always the un-perturbed decode, so the
    // operator can only pick something other than the plain policy's answer when
    // the value head prefers it.
    int lookahead_k = 0;

    // Gumbel noise added to the buy scores to generate candidates 1..k-1. The
    // buy decoder is a threshold-and-sort with no sampling in it, which is why
    // `temperature` alone never diversified buy at all - perturbing the scores
    // is what produces genuinely different purchase sets.
    float lookahead_tau = 1.0f;

    // ABLATION. After the lookahead picks a purchase list, throw away its
    // QUANTITIES and rebuild the list from its distinct (hex, variant) cells
    // with the ordinary repeat-until-broke rule.
    //
    // That is exactly the information the training label can carry - the buy
    // target is a bitmask over cells, so a list that buys four infantry and one
    // that buys six are the same label. Measured, 12.0% of all buy decisions
    // are changed by the lookahead in quantity ONLY, against 10.2% changed in
    // which cells. This flag measures how much of the operator's strength
    // survives that restriction, which is the ceiling on what cloning it can
    // ever reach.
    bool lookahead_cells_only = false;
};

// Always 0 now: the architecture lives in train/model.py, so a parameter count
// cannot be derived from a config. neural_describe reports the loaded module's.
long long neural_param_count(const NeuralConfig& cfg);

// Builds a policy with randomly initialised weights. Returns null and fills
// `error` if LibTorch was not compiled in, or CUDA was requested and is absent.
std::unique_ptr<BatchedPolicy> make_neural_policy(const NeuralConfig& cfg, std::string& error);

// True when this build has LibTorch. Lets callers give a useful message rather
// than a link error.
bool neural_available();

// A short description of the built model - device, dtype, parameter count.
std::string neural_describe(const BatchedPolicy& policy);

}  // namespace oo
