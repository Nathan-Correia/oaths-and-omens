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
    // Architecture (§10.10). d=128 x 4 blocks is the recommended starting point:
    // ~0.9M parameters, and below d=128 tensor cores stop being efficient, so a
    // smaller model buys far less speed than its FLOP count suggests.
    int d_model = 128;
    int n_blocks = 4;
    int n_heads = 4;
    int ffn_mult = 4;
    int stem_layers = 2;  // hex-conv layers before the transformer (§10.6)

    // The bucket this instance serves.
    int radius = 7;
    int num_factions = 8;

    bool cuda = true;
    bool fp16 = true;

    // Seed for weight initialisation. M9 runs on random weights; M10 loads real
    // ones instead.
    int64_t seed = 1;
};

// Number of parameters a config implies, without building it. Handy for
// sanity-checking a size against §10.10's table before paying for the model.
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
