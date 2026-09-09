// The no-LibTorch build of the neural policy (PLAN.md §10).
//
// Compiled instead of neural.cpp when OO_LIBTORCH_DIR is unset, so every caller
// links either way and the failure is a clear message at run time rather than an
// unresolved symbol at link time. §1.2's rule, applied to a GPU dependency: the
// core never requires anything external.

#include "oo/neural.hpp"

namespace oo {

bool neural_available() { return false; }

long long neural_param_count(const NeuralConfig&) { return 0; }

std::unique_ptr<BatchedPolicy> make_neural_policy(const NeuralConfig&, std::string& error) {
    error =
        "this build has no LibTorch; reconfigure with -DOO_LIBTORCH_DIR=<path> "
        "to enable the neural policy";
    return nullptr;
}

std::string neural_describe(const BatchedPolicy&) { return "(no LibTorch in this build)"; }

}  // namespace oo
