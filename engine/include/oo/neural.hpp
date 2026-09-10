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
