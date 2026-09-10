// The neural policy, in LibTorch (PLAN.md §10, M9c).
//
// Only compiled when OO_LIBTORCH_DIR is set; see neural_stub.cpp for the other
// build. Nothing else in the engine includes a torch header.

#include "oo/neural.hpp"

#include "oo/buy.hpp"
#include "oo/encode.hpp"
#include "oo/grid.hpp"

#include <torch/script.h>
#include <torch/torch.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <sstream>
#include <vector>

namespace oo {

namespace {

namespace F = torch::nn::functional;

// The record the module consumes. Its LAYOUT is not described here any more -
// train/model.py owns that, and the module we load does the expansion itself.
// That is the point of loading TorchScript rather than rebuilding the network in
// C++: there was previously a second definition of every feature, and a mismatch
// between them does not raise, it just feeds the net misaligned inputs.
constexpr int kPosBytes = static_cast<int>(sizeof(EncodedPosition));

// Per-token feature widths after expansion (§10.4).
constexpr int kHexFeatures = 54;
constexpr int kFactionFeatures = 15;  // active, is_me, outposts, 3 units, settle, gold, 4 resources, vp, kill_xp, vp_behind
constexpr int kHeaderFeatures = 15;

// Relative-distance bias buckets (§10.5). Nine for hex-to-hex distance, then
// one each for the token pairs that have no hex metric at all.
constexpr int kDistBuckets = 9;
constexpr int kBucketHexFaction = 9;
constexpr int kBucketHexCls = 10;
constexpr int kBucketFactionFaction = 11;
constexpr int kBucketFactionCls = 12;
constexpr int kBucketClsCls = 13;
constexpr int kNumBuckets = 14;

// Log bucketing: care a lot about 1 vs 2, almost nothing about 15 vs 16.
int distance_bucket(int d) {
    if (d <= 5) return d;
    if (d <= 7) return 6;
    if (d <= 11) return 7;
    return 8;
}

// Reassembles a little-endian int16 from two uint8 lanes, sign-extended.
torch::Tensor read_i16(const torch::Tensor& bytes, int lo) {
    auto v = bytes.select(-1, lo) + bytes.select(-1, lo + 1) * 256;
    // Values above 32767 are negative; wrap them rather than clamping, which
    // would silently turn a deficit into a huge positive.
    return torch::where(v > 32767, v - 65536, v);
}

torch::Tensor bit(const torch::Tensor& v, int k) {
    return torch::bitwise_and(torch::bitwise_right_shift(v, k), 1);
}

}  // namespace

// --- the policy -------------------------------------------------------------

namespace {

class TorchPolicy : public BatchedPolicy {
public:
    TorchPolicy(const NeuralConfig& cfg, torch::jit::script::Module module, int hexes,
                long long params)
        : cfg_(cfg),
          hexes_(hexes),
          tokens_(hexes + cfg.num_factions + 1),
          device_(cfg.cuda ? torch::kCUDA : torch::kCPU),
          module_(std::move(module)),
          params_(params) {
        module_.to(device_);
        // Deliberately NO dtype conversion here. Module::to(kFloat16) converts
        // every tensor in the module including the integer geometry buffers, and
        // a half index tensor fails inside gather. Precision is chosen at export
        // time instead (train/export.py --half), where .half() touches only
        // floating-point tensors.
        module_.eval();

        for (const auto& p : module_.parameters()) {
            half_ = p.scalar_type() == torch::kFloat16;
            break;
        }
    }

    void evaluate(const DecisionRequest* requests, DecisionAnswer* answers, int n) override;
    void evaluate_impl(const DecisionRequest* requests, DecisionAnswer* answers, int n);

    std::string describe() const {
        std::ostringstream os;
        os << tokens_ << " tokens, " << (params_ / 1000) << "k params, "
           << (cfg_.cuda ? "cuda" : "cpu") << (half_ ? "/fp16" : "/fp32") << ", "
           << cfg_.weights;
        return os.str();
    }

private:
    NeuralConfig cfg_;
    int hexes_, tokens_;
    torch::Device device_;
    torch::jit::script::Module module_;
    long long params_ = 0;
    bool half_ = false;
    mutable std::mutex gpu_mu_;
};

void TorchPolicy::evaluate(const DecisionRequest* requests, DecisionAnswer* answers, int n) {
    // A torch error thrown on a game thread would otherwise propagate out of the
    // thread body and terminate the process through __fastfail - which on Windows
    // means exit code 0xC0000409 and NOT ONE BYTE of buffered stdout, so the
    // banner printed seconds earlier vanishes too. Catching it here turns a
    // silent kill into a message naming the op that failed.
    try {
        evaluate_impl(requests, answers, n);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "neural policy failed on a batch of %d: %s\n", n, e.what());
        std::fflush(stderr);
        std::abort();
    }
}

void TorchPolicy::evaluate_impl(const DecisionRequest* requests, DecisionAnswer* answers, int n) {
    torch::NoGradGuard no_grad;

    // THREAD_LOCAL, not a member. BatchGate runs evaluate() outside its lock so
    // games can keep arriving, which means two threads can be in here at once -
    // a shared staging buffer is a straight data race, and it announces itself
    // as a stack-buffer-overrun crash rather than anything legible.
    // PINNED, when there is a device to talk to. A copy out of ordinary pageable
    // memory has to be staged through a driver bounce buffer first; pinned pages
    // can be DMA'd straight across, which is worth roughly 2x on both directions.
    // Reallocated only when the batch size changes, which after warm-up it does
    // not.
    thread_local torch::Tensor staging;
    thread_local std::vector<EncodedPosition> staging_fallback;
    EncodedPosition* dst = nullptr;
    if (cfg_.cuda) {
        if (!staging.defined() || staging.size(0) != n) {
            staging = torch::empty(
                {n, kPosBytes},
                torch::TensorOptions().dtype(torch::kUInt8).pinned_memory(true));
        }
        dst = reinterpret_cast<EncodedPosition*>(staging.data_ptr<uint8_t>());
    } else {
        staging_fallback.resize(static_cast<size_t>(n));
        dst = staging_fallback.data();
    }
    encode_batch(requests, n, dst);

    auto raw = cfg_.cuda
                   ? staging.to(device_, /*non_blocking=*/false)
                   : torch::from_blob(dst, {n, kPosBytes}, torch::kUInt8).clone();

    // One batch on the GPU at a time. Encoding above is parallel and worth keeping
    // so; overlapping the forward passes themselves would only multiply
    // working-set memory for no throughput, since the device is the bottleneck.
    //
    // SCOPED to the device work only. Holding it across the decode loop below
    // serialised that too, and the decode is pure CPU work over a thread-local
    // buffer - several milliseconds per batch of argmaxes and sorts that has no
    // reason to be single-file.
    std::unique_lock<std::mutex> gpu_lock(gpu_mu_);

    // Two heads ask about ONE hex - the query hex carried in the request - so
    // gather that row per position before running them. Computing head_rect over
    // every hex made it [B, n_hex, 48], which at batch 296 is 9.6 MB, 72 % of all
    // download traffic, of which ~99 % was discarded. Same numbers, 160x less of
    // them.
    std::vector<int64_t> query(static_cast<size_t>(n));
    for (int i = 0; i < n; ++i) {
        const int h = requests[i].hex;
        query[static_cast<size_t>(i)] =
            (h >= 0 && h < requests[i].state->num_hexes) ? h : 0;
    }
    auto query_idx = torch::from_blob(query.data(), {n}, torch::kLong).to(device_);

    // ONE forward, ONE download. The module gathers the query hex itself and
    // returns every head as a dict; concatenating on the device and copying once
    // costs a single synchronisation, where pulling the heads back separately
    // serialised the batch once per head.
    //
    // `move` already carries its pass logit as a trailing column
    // (train/model.py), so it is hexes*6 + 1 wide.
    const int nf = cfg_.num_factions;
    auto out = module_.forward({raw, query_idx}).toGenericDict();
    auto head = [&](const char* name) { return out.at(name).toTensor(); };
    auto flat = torch::cat({head("move"), head("hex"), head("buy"), head("target"),
                            head("resource"), head("swap"), head("rect")},
                           1);
    auto device_f32 = flat.to(torch::kFloat32);
    thread_local torch::Tensor host_pinned;
    torch::Tensor host;
    if (cfg_.cuda) {
        if (!host_pinned.defined() || host_pinned.sizes() != device_f32.sizes()) {
            host_pinned = torch::empty(
                device_f32.sizes(),
                torch::TensorOptions().dtype(torch::kFloat32).pinned_memory(true));
        }
        host_pinned.copy_(device_f32);
        host = host_pinned;
    } else {
        host = device_f32.to(torch::kCPU);
    }
    // The copy above is complete, and `host` is this thread's own buffer, so the
    // device is free from here on.
    gpu_lock.unlock();

    const float* base = host.data_ptr<float>();
    const int64_t row = host.size(1);

    // Column offsets into each row of the fused result.
    const int64_t off_move = 0;
    const int64_t off_pass = off_move + int64_t(hexes_) * NUM_DIRECTIONS;
    const int64_t off_hex = off_pass + 1;
    const int64_t off_buy = off_hex + hexes_;
    const int64_t off_target = off_buy + int64_t(hexes_) * 9;
    const int64_t off_resource = off_target + nf;
    const int64_t off_swap = off_resource + 2;
    const int64_t off_rect = off_swap + 2;

    auto move_at = [&](int i, int h, int d) { return base[i * row + off_move + h * NUM_DIRECTIONS + d]; };
    auto pass_at = [&](int i) { return base[i * row + off_pass]; };
    auto hex_at = [&](int i, int h) { return base[i * row + off_hex + h]; };
    auto buy_at = [&](int i, int h, int v) { return base[i * row + off_buy + h * 9 + v]; };
    auto target_at = [&](int i, int f) { return base[i * row + off_target + f]; };
    auto res_at = [&](int i, int k) { return base[i * row + off_resource + k]; };
    auto swap_at = [&](int i, int k) { return base[i * row + off_swap + k]; };
    auto rect_at = [&](int i, int k) { return base[i * row + off_rect + k]; };

    for (int i = 0; i < n; ++i) {
        const DecisionRequest& r = requests[i];
        DecisionAnswer& a = answers[i];
        const GameState& s = *r.state;

        switch (r.kind) {
            case DecisionKind::kMovement:
            case DecisionKind::kCavalry: {
                // Legality masking is not optional (§10.7): the argmax runs over
                // legal cells only, and `pass` competes with them.
                float best = pass_at(i);
                int best_h = -1, best_d = -1;
                for (int h = 0; h < s.num_hexes; ++h) {
                    for (int dir = 0; dir < NUM_DIRECTIONS; ++dir) {
                        if (!r.legal_mask->cell[h][dir]) continue;
                        const float v = move_at(i, h, dir);
                        if (v > best) {
                            best = v;
                            best_h = h;
                            best_d = dir;
                        }
                    }
                }
                a.flag = best_h >= 0;
                a.move.hex = static_cast<int16_t>(best_h);
                a.move.dir = static_cast<int8_t>(best_d);
                break;
            }
            case DecisionKind::kPlacement: {
                int best_h = -1;
                float best = 0.0f;
                for (int h = 0; h < s.num_hexes; ++h) {
                    if (!r.legal_hexes[h]) continue;
                    const float v = hex_at(i, h);
                    if (best_h < 0 || v > best) {
                        best = v;
                        best_h = h;
                    }
                }
                a.index = best_h;
                break;
            }
            case DecisionKind::kDraft: {
                int best_k = 0;
                float best = 0.0f;
                for (int k = 0; k < r.pool_size; ++k) {
                    const float v = hex_at(i, r.pool[k]);
                    if (k == 0 || v > best) {
                        best = v;
                        best_k = k;
                    }
                }
                a.index = r.pool_size > 0 ? r.pool[best_k] : -1;
                break;
            }
            case DecisionKind::kTarget: {
                // Only factions actually in this battle are valid targets.
                const Battle* b = s.battle_at(r.hex);
                int best_f = -1;
                float best = 0.0f;
                if (b) {
                    for (int slot = 0; slot < b->nslots; ++slot) {
                        const int f = b->slots[slot].faction;
                        if (f == NO_FACTION || f == r.faction) continue;
                        const int rel = (f - r.faction + s.num_factions) % s.num_factions;
                        const float v = target_at(i, rel);
                        if (best_f < 0 || v > best) {
                            best = v;
                            best_f = f;
                        }
                    }
                }
                a.index = best_f;
                break;
            }
            case DecisionKind::kResource:
                // Already gathered at the query hex, so there is no hex index here.
                a.resource = res_at(i, 0) > res_at(i, 1) ? kIron : kFish;
                break;
            case DecisionKind::kSwap:
                a.flag = swap_at(i, 0) > swap_at(i, 1);
                break;
            case DecisionKind::kBuy: {
                // Score every legal action by its (hex, variant) cell and take
                // them in score order while positive. Actions that turn out to be
                // unaffordable are silently skipped by apply_buy_phase, so a
                // greedy list cannot produce an illegal state.
                a.buy->clear();
                struct Scored {
                    float score;
                    int index;
                };
                std::vector<Scored> scored;
                scored.reserve(static_cast<size_t>(r.legal_buy->size()));
                for (int k = 0; k < r.legal_buy->size(); ++k) {
                    const BuyAction& act = (*r.legal_buy)[k];
                    int variant = 0;
                    switch (act.type) {
                        case BuyType::kBuyInfantry: variant = 0; break;
                        case BuyType::kConvertToSpecial: variant = 1 + act.unit_type; break;
                        case BuyType::kBuildOutpost: variant = 3 + act.unit_type; break;
                        default: variant = 6 + act.upgrade; break;
                    }
                    if (variant > 8) variant = 8;
                    scored.push_back({buy_at(i, act.hex, variant), k});
                }
                std::stable_sort(scored.begin(), scored.end(),
                                 [](const Scored& p, const Scored& q) { return p.score > q.score; });
                for (const Scored& sc : scored) {
                    if (sc.score <= 0.0f) break;
                    if (a.buy->size() >= kMaxChosenBuy) break;
                    a.buy->push_back((*r.legal_buy)[sc.index]);
                }
                break;
            }
            case DecisionKind::kRectification: {
                // Score (battle slot, unit type) and send units back one at a
                // time until the stack is under the cap.
                a.send_back->clear();
                const Battle* b = s.battle_at(r.hex);
                if (!b) break;
                int have[MAX_BATTLE_CONTRIB][NUM_UNIT_TYPES] = {};
                int total = 0;
                for (int slot = 0; slot < b->nslots; ++slot) {
                    if (b->slots[slot].faction != r.winner) continue;
                    for (int t = 0; t < NUM_UNIT_TYPES; ++t) {
                        have[slot][t] = b->slots[slot].units[t];
                        total += have[slot][t];
                    }
                }
                int sent[MAX_BATTLE_CONTRIB][NUM_UNIT_TYPES] = {};
                while (total > r.cap) {
                    int bs = -1, bt = -1;
                    float best = 0.0f;
                    for (int slot = 0; slot < b->nslots; ++slot) {
                        for (int t = 0; t < NUM_UNIT_TYPES; ++t) {
                            if (have[slot][t] <= 0) continue;
                            const float v = rect_at(i, slot * NUM_UNIT_TYPES + t);
                            if (bs < 0 || v > best) {
                                best = v;
                                bs = slot;
                                bt = t;
                            }
                        }
                    }
                    if (bs < 0) break;
                    --have[bs][bt];
                    ++sent[bs][bt];
                    --total;
                }
                for (int slot = 0; slot < b->nslots; ++slot) {
                    int any = 0;
                    for (int t = 0; t < NUM_UNIT_TYPES; ++t) any += sent[slot][t];
                    if (!any) continue;
                    SendBackEntry e{};
                    e.origin_hex = b->slots[slot].origin;
                    for (int t = 0; t < NUM_UNIT_TYPES; ++t) {
                        e.units[t] = static_cast<int16_t>(sent[slot][t]);
                    }
                    a.send_back->push_back(e);
                }
                break;
            }
        }
    }
}

}  // namespace

bool neural_available() { return true; }

long long neural_param_count(const NeuralConfig&) {
    // Architecture lives in train/model.py now, so this cannot be derived from a
    // config any more. It is reported from the loaded module instead - see
    // neural_describe.
    return 0;
}

std::unique_ptr<BatchedPolicy> make_neural_policy(const NeuralConfig& cfg, std::string& error) {
    if (cfg.weights.empty()) {
        error =
            "no weights given. The network is defined in train/model.py and loaded as "
"TorchScript; export one with:\n"
            "  python train/export.py --init --out checkpoints/init.ts "
            "--radius <r> --factions <f>";
        return nullptr;
    }
    if (cfg.cuda && !torch::cuda::is_available()) {
        error = "CUDA requested but not available to LibTorch";
        return nullptr;
    }

    torch::jit::script::Module module;
    try {
        module = torch::jit::load(cfg.weights);
    } catch (const std::exception& e) {
        error = "cannot load " + cfg.weights + ": " + e.what();
        return nullptr;
    }

    // The exporter stamps these so a checkpoint can be refused rather than
    // quietly played against the wrong board. A radius mismatch would reshape
    // every hex token; an encoding mismatch would feed misaligned features. Both
    // produce a net that plays like noise and nothing that looks like an error.
    auto attr_int = [&](const char* name, int fallback) {
        try {
            return static_cast<int>(module.attr(name).toInt());
        } catch (const std::exception&) {
            return fallback;
        }
    };
    const int radius = attr_int("radius", -1);
    const int factions = attr_int("num_factions", -1);
    const int enc = attr_int("encoding_version", -1);

    if (enc >= 0 && enc != kEncodingVersion) {
        error = "checkpoint was trained against encoding version " + std::to_string(enc) +
                ", engine writes version " + std::to_string(kEncodingVersion);
        return nullptr;
    }
    if (radius >= 0 && radius != cfg.radius) {
        error = "checkpoint is for radius " + std::to_string(radius) + ", playing radius " +
                std::to_string(cfg.radius);
        return nullptr;
    }
    if (factions >= 0 && factions != cfg.num_factions) {
        error = "checkpoint is for " + std::to_string(factions) + " factions, playing " +
                std::to_string(cfg.num_factions);
        return nullptr;
    }

    long long params = 0;
    for (const auto& p : module.parameters()) params += p.numel();

    return std::make_unique<TorchPolicy>(cfg, std::move(module),
                                         HexGrid::shared(cfg.radius).num_hexes(), params);
}

std::string neural_describe(const BatchedPolicy& policy) {
    const auto* p = dynamic_cast<const TorchPolicy*>(&policy);
    return p ? p->describe() : std::string("(not a neural policy)");
}

}  // namespace oo
