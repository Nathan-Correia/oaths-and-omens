// The neural policy, in LibTorch (PLAN.md §10, M9c).
//
// Only compiled when OO_LIBTORCH_DIR is set; see neural_stub.cpp for the other
// build. Nothing else in the engine includes a torch header.

#include "oo/neural.hpp"

#include "oo/buy.hpp"
#include "oo/encode.hpp"
#include "oo/grid.hpp"
#include "oo/rng.hpp"

#include <torch/script.h>
#include <torch/torch.h>

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <memory>
#include <mutex>
#include <thread>
#include <utility>
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

// Lookahead diagnostics. The question these answer: when the operator departs
// from the plain decode, is the difference something the TRAINING LABEL can
// express? The buy label is a bitmask over (hex, variant) cells, so a change of
// quantity - four infantry instead of six at the same city - collapses to the
// same bits and cannot be cloned at all. Printed on destruction.
std::atomic<long long> g_la_decisions{0};
std::atomic<long long> g_la_changed{0};
std::atomic<long long> g_la_cells_differ{0};
std::atomic<long long> g_la_qty_only{0};

// How many extra copies of each repeatable buy action to propose. Capitals are
// exempt from the per-turn recruit cap (buy.cpp), so the real limit is gold, and
// gold accumulates over a game - 32 was enough for the starting 50 but would
// quietly cap a rich late-game turn. The engine drops every surplus action, so
// over-proposing costs a little list-building and nothing else; under-proposing
// silently leaves money unspent.
constexpr int kBuyRepeatPasses = 128;

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

    // Per-thread sampling stream, only touched when temperature > 0.
    thread_local Rng sample_rng(cfg_.sample_seed +
                                static_cast<int64_t>(std::hash<std::thread::id>()(
                                    std::this_thread::get_id())));

    // Picks among (index, score) candidates: the best one at temperature 0, a
    // softmax draw above it.
    auto choose = [&](const std::vector<std::pair<int, float> >& cand) -> int {
        if (cand.empty()) return -1;
        if (cfg_.temperature <= 0.0f) {
            int best = cand[0].first;
            float bs = cand[0].second;
            for (size_t i = 1; i < cand.size(); ++i) {
                if (cand[i].second > bs) {
                    bs = cand[i].second;
                    best = cand[i].first;
                }
            }
            return best;
        }
        // Shift by the max before exponentiating. Raw logits reach magnitudes
        // where exp() overflows to inf and every weight becomes nan.
        float mx = cand[0].second;
        for (size_t i = 1; i < cand.size(); ++i) {
            if (cand[i].second > mx) mx = cand[i].second;
        }
        std::vector<double> w(cand.size());
        double total = 0.0;
        for (size_t i = 0; i < cand.size(); ++i) {
            w[i] = std::exp(double(cand[i].second - mx) / double(cfg_.temperature));
            total += w[i];
        }
        double r = sample_rng.random() * total;
        for (size_t i = 0; i < cand.size(); ++i) {
            r -= w[i];
            if (r <= 0.0) return cand[i].first;
        }
        return cand.back().first;
    };

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

    // The buy decode, hoisted so the lookahead below can run it repeatedly with
    // different noise to produce candidate purchase sets. `tau` 0 reproduces the
    // plain deterministic decode exactly.
    // The buy decode, hoisted so the lookahead below can run it repeatedly to
    // produce candidate purchase sets.
    //
    // `delta` SWAPS ONE MARGINAL CHOICE, and is the whole reason candidates are
    // generated this way rather than by sampling. The first attempt drew
    // candidates with Gumbel noise, which made the teacher a RANDOM function of
    // the state: it answered differently for the same position, so cloning it
    // learned the expectation of best-of-k noise, which is close to the plain
    // policy. Measured, the cloned net reproduced the teacher's decoded set
    // 88.5% of the time against the generating net's own 89.0% - it learned
    // nothing, because the selection effect is precisely what averaging
    // destroys.
    //
    // Shifting the margin instead is deterministic: candidate j is a fixed
    // function of the position, so a network CAN represent the operator's
    // answer. It also aims the search where a value head is actually useful -
    // the cells right at the threshold, where the policy head is least sure
    // whether the purchase is worth it.
    auto decode_buy = [&](int i, const DecisionRequest& r, ChosenBuyActions& out,
                          int delta) {
        out.clear();
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
        if (scored.empty()) return;

        // WHICH actions to take: the top T by score, where T is normally the
        // number scoring above zero. Sorting by score alone here is right
        // because the threshold is a score question; the type ordering below is
        // a separate, spending-order question.
        std::vector<Scored> by_score = scored;
        std::stable_sort(by_score.begin(), by_score.end(),
                         [](const Scored& p, const Scored& q) { return p.score > q.score; });
        int n_pass = 0;
        while (n_pass < static_cast<int>(by_score.size()) &&
               by_score[static_cast<size_t>(n_pass)].score > 0.0f) {
            ++n_pass;
        }
        int take = n_pass;
        if (take > kMaxChosenBuy) take = kMaxChosenBuy;

        // Candidate `delta` SWAPS the delta-th weakest selected action for the
        // delta-th strongest rejected one. The count is preserved exactly, and
        // that is not a detail - it is the constraint the whole operator lives
        // under.
        //
        // The first deterministic attempt shifted the threshold instead, so
        // candidates differed in HOW MANY purchases they made. That collapsed
        // play to 15.6% against the same weights without lookahead, worsening
        // monotonically with k (11.9% at k=9) and scoring 49 VP against 136:
        // the operator had learned to buy nothing. The value head is trained
        // only on decision-point states, where the turn's gold is still unspent,
        // so it cannot rank afterstates that differ in total spending - offered
        // the choice it prefers the full treasury every time. Holding the count
        // fixed keeps every candidate on the distribution V was actually
        // trained on, and asks it only the question it can answer: given that
        // this much is being spent, on WHAT should it go.
        const int n_all = static_cast<int>(by_score.size());
        if (delta > 0 && take - delta >= 0 && take - 1 + delta < n_all) {
            std::swap(by_score[static_cast<size_t>(take - delta)],
                      by_score[static_cast<size_t>(take - 1 + delta)]);
        }
        by_score.resize(static_cast<size_t>(take));

        // IN WHAT ORDER to spend: by TYPE first, score second.
        //
        // apply_buy_phase spends gold in list order, and an outpost is the only
        // source of victory points in the game (collect.cpp: only non-capital
        // cities score). Emitting in pure score order let infantry - 25 of them
        // at 2 gold each - drain the treasury before the 3-gold outpost was ever
        // attempted, so the policy could hold territory and still score nothing.
        //
        // The teacher orders the same way and for the same reason:
        // greedy_buy_common pushes its one upgrade-or-outpost action before any
        // infantry or conversions. Upgrades come before new outposts because
        // only one of the two happens per turn, and improving ground already
        // held beats founding more.
        auto rank = [](BuyType t) {
            switch (t) {
                case BuyType::kUpgradeOutpost: return 0;
                case BuyType::kBuildOutpost: return 1;
                case BuyType::kBuyInfantry: return 2;
                default: return 3;  // conversions last
            }
        };
        std::stable_sort(by_score.begin(), by_score.end(),
                         [&](const Scored& p, const Scored& q) {
                             const int rp = rank((*r.legal_buy)[p.index].type);
                             const int rq = rank((*r.legal_buy)[q.index].type);
                             if (rp != rq) return rp < rq;
                             return p.score > q.score;
                         });
        int n_selected = 0;
        for (const Scored& sc : by_score) {
            if (out.size() >= kMaxChosenBuy) break;
            out.push_back((*r.legal_buy)[sc.index]);
            ++n_selected;
        }

        // REPEAT the repeatable picks until the budget stops them.
        //
        // The teacher buys `gold / kInfantryCost` infantry - 25 of them on turn
        // one - by pushing the same action over and over. The training label is
        // a bitmask with one bit per (hex, variant), so all 25 collapse into one
        // bit and the net is taught "buy infantry here" ONCE. Emitting each cell
        // once then made it structurally impossible to buy more than one unit
        // per city per turn, which starved it of units. Measured: with buy
        // handed to the teacher instead, win rate against greedy went 18% ->
        // 69%.
        //
        // Repeating is safe because apply_one validates every action and
        // silently skips what cannot be afforded (buy.cpp), so the engine - not
        // this decoder - decides where the money runs out. Only infantry and
        // conversions repeat; the per-turn cap allows just one outpost or
        // upgrade action anyway.
        for (int pass = 0; pass < kBuyRepeatPasses && n_selected > 0; ++pass) {
            bool appended = false;
            for (int k = 0; k < n_selected; ++k) {
                const BuyAction& act = out[k];
                if (act.type != BuyType::kBuyInfantry &&
                    act.type != BuyType::kConvertToSpecial) {
                    continue;
                }
                if (out.size() >= kMaxChosenBuy) break;
                out.push_back(act);
                appended = true;
            }
            if (!appended) break;
        }
    };

    for (int i = 0; i < n; ++i) {
        const DecisionRequest& r = requests[i];
        DecisionAnswer& a = answers[i];
        const GameState& s = *r.state;

        switch (r.kind) {
            case DecisionKind::kMovement:
            case DecisionKind::kCavalry: {
                // Legality masking is not optional (§10.7): the argmax runs over
                // legal cells only, and `pass` competes with them.
                // Encoded hex*6+dir, with num_hexes*6 meaning "pass" - the same
                // layout the training labels use.
                std::vector<std::pair<int, float> > cand;
                cand.push_back(std::make_pair(s.num_hexes * NUM_DIRECTIONS, pass_at(i)));
                for (int h = 0; h < s.num_hexes; ++h) {
                    for (int dir = 0; dir < NUM_DIRECTIONS; ++dir) {
                        if (!r.legal_mask->cell[h][dir]) continue;
                        cand.push_back(
                            std::make_pair(h * NUM_DIRECTIONS + dir, move_at(i, h, dir)));
                    }
                }
                const int pick = choose(cand);
                a.flag = pick >= 0 && pick < s.num_hexes * NUM_DIRECTIONS;
                a.move.hex = static_cast<int16_t>(a.flag ? pick / NUM_DIRECTIONS : -1);
                a.move.dir = static_cast<int8_t>(a.flag ? pick % NUM_DIRECTIONS : -1);
                break;
            }
            case DecisionKind::kPlacement: {
                std::vector<std::pair<int, float> > cand;
                for (int h = 0; h < s.num_hexes; ++h) {
                    if (r.legal_hexes[h]) cand.push_back(std::make_pair(h, hex_at(i, h)));
                }
                a.index = choose(cand);
                break;
            }
            case DecisionKind::kDraft: {
                std::vector<std::pair<int, float> > cand;
                for (int k = 0; k < r.pool_size; ++k) {
                    cand.push_back(std::make_pair(int(r.pool[k]), hex_at(i, r.pool[k])));
                }
                a.index = choose(cand);
                break;
            }
            case DecisionKind::kTarget: {
                // Only factions actually in this battle are valid targets.
                const Battle* b = s.battle_at(r.hex);
                std::vector<std::pair<int, float> > cand;
                if (b) {
                    for (int slot = 0; slot < b->nslots; ++slot) {
                        const int f = b->slots[slot].faction;
                        if (f == NO_FACTION || f == r.faction) continue;
                        const int rel = (f - r.faction + s.num_factions) % s.num_factions;
                        cand.push_back(std::make_pair(f, target_at(i, rel)));
                    }
                }
                a.index = choose(cand);
                break;
            }
            case DecisionKind::kResource:
                // Already gathered at the query hex, so there is no hex index here.
                a.resource = res_at(i, 0) > res_at(i, 1) ? kIron : kFish;
                break;
            case DecisionKind::kSwap:
                a.flag = swap_at(i, 0) > swap_at(i, 1);
                break;
            case DecisionKind::kBuy:
                decode_buy(i, r, *a.buy, 0);
                break;
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

    // --- ONE-PLY VALUE LOOKAHEAD ON BUY (see NeuralConfig::lookahead_k) -----
    //
    // Everything above is the plain policy. This is the part that makes the
    // answer BETTER than the policy that proposed it, which is the property a
    // self-play loop needs and the one three previous objectives lacked.
    if (cfg_.lookahead_k > 1) {
        std::vector<int> buy_idx;
        for (int i = 0; i < n; ++i) {
            if (requests[i].kind == DecisionKind::kBuy) buy_idx.push_back(i);
        }
        if (!buy_idx.empty()) {
            const int k = cfg_.lookahead_k;
            const size_t slots = buy_idx.size();

            auto same_buy = [](const ChosenBuyActions& x, const ChosenBuyActions& y) {
                if (x.size() != y.size()) return false;
                for (int t = 0; t < x.size(); ++t) {
                    if (!(x[t] == y[t])) return false;
                }
                return true;
            };

            // Candidate 0 is always the un-perturbed decode, so the operator can
            // only depart from the plain policy when the value head says to.
            std::vector<ChosenBuyActions> cand(slots * static_cast<size_t>(k));
            std::vector<EncodedPosition> succ;
            std::vector<int> owner, which;
            succ.reserve(slots * static_cast<size_t>(k));

            // A GameState is far too big for the stack, and this runs on a game
            // thread that is already deep in a turn. One scratch copy is enough:
            // each successor is encoded immediately and the copy reused.
            std::unique_ptr<GameState> scratch(new GameState());
            std::vector<ChosenBuyActions> per_faction(MAX_FACTIONS);

            for (size_t si = 0; si < slots; ++si) {
                const int i = buy_idx[si];
                const DecisionRequest& r = requests[i];
                if (r.faction < 0 || r.faction >= MAX_FACTIONS) continue;
                for (int j = 0; j < k; ++j) {
                    ChosenBuyActions& c = cand[si * static_cast<size_t>(k) + j];
                    // Candidate 0 is the plain decode; j>0 swaps the j-th
                    // marginal selection for the j-th best rejected one, so the
                    // operator can only depart from the policy when the value
                    // head prefers a DIFFERENT purchase, never a cheaper turn.
                    decode_buy(i, r, c, j);
                    // Gumbel noise reproduces the plain decode often enough that
                    // skipping duplicates is most of the batch back.
                    bool dup = false;
                    for (int p = 0; p < j && !dup; ++p) {
                        dup = same_buy(cand[si * static_cast<size_t>(k) + p], c);
                    }
                    if (dup) continue;

                    *scratch = *r.state;
                    for (int f = 0; f < MAX_FACTIONS; ++f) per_faction[f].clear();
                    per_faction[r.faction] = c;
                    apply_buy_phase(*scratch, per_faction.data());

                    DecisionRequest sr = r;
                    sr.state = scratch.get();
                    succ.emplace_back();
                    encode_position(*scratch, sr, succ.back());
                    owner.push_back(static_cast<int>(si));
                    which.push_back(j);
                }
            }

            const int m = static_cast<int>(succ.size());
            if (m > 0) {
                std::vector<int64_t> q(static_cast<size_t>(m), 0);
                auto in = torch::from_blob(succ.data(), {m, kPosBytes}, torch::kUInt8);

                std::unique_lock<std::mutex> lk(gpu_mu_);
                auto raw2 = in.to(device_);
                auto q2 = torch::from_blob(q.data(), {m}, torch::kLong).to(device_);
                auto o2 = module_.forward({raw2, q2}).toGenericDict();
                auto v = o2.at("value").toTensor().to(torch::kFloat32).to(torch::kCPU);
                lk.unlock();

                const float* vp = v.data_ptr<float>();
                std::vector<float> best(slots, -3.0e38f);
                std::vector<int> best_j(slots, 0);
                for (int t = 0; t < m; ++t) {
                    const int si = owner[static_cast<size_t>(t)];
                    if (vp[t] > best[static_cast<size_t>(si)]) {
                        best[static_cast<size_t>(si)] = vp[t];
                        best_j[static_cast<size_t>(si)] = which[static_cast<size_t>(t)];
                    }
                }
                // The set of (hex, variant) cells a purchase list touches -
                // exactly what the training bitmask records, and nothing more.
                auto cell_set = [](const ChosenBuyActions& c) {
                    std::vector<int> v;
                    for (int t = 0; t < c.size(); ++t) {
                        int variant = 0;
                        switch (c[t].type) {
                            case BuyType::kBuyInfantry: variant = 0; break;
                            case BuyType::kConvertToSpecial: variant = 1 + c[t].unit_type; break;
                            case BuyType::kBuildOutpost: variant = 3 + c[t].unit_type; break;
                            default: variant = 6 + c[t].upgrade; break;
                        }
                        if (variant > 8) variant = 8;
                        v.push_back(c[t].hex * 9 + variant);
                    }
                    std::sort(v.begin(), v.end());
                    v.erase(std::unique(v.begin(), v.end()), v.end());
                    return v;
                };
                for (size_t si = 0; si < slots; ++si) {
                    ChosenBuyActions* dst_buy = answers[buy_idx[si]].buy;
                    const ChosenBuyActions& plain = cand[si * static_cast<size_t>(k)];
                    const ChosenBuyActions& pick =
                        cand[si * static_cast<size_t>(k) + static_cast<size_t>(best_j[si])];
                    g_la_decisions.fetch_add(1, std::memory_order_relaxed);
                    if (best_j[si] != 0) {
                        g_la_changed.fetch_add(1, std::memory_order_relaxed);
                        if (cell_set(plain) != cell_set(pick)) {
                            g_la_cells_differ.fetch_add(1, std::memory_order_relaxed);
                        } else {
                            g_la_qty_only.fetch_add(1, std::memory_order_relaxed);
                        }
                    }
                    if (!dst_buy) continue;
                    if (!cfg_.lookahead_cells_only) {
                        *dst_buy = pick;
                        continue;
                    }
                    // Keep only what a bitmask label could reproduce: the
                    // distinct cells, in the decoder's type order, re-expanded
                    // by the ordinary repeat rule.
                    ChosenBuyActions reduced;
                    reduced.clear();
                    std::vector<int> seen;
                    for (int t = 0; t < pick.size(); ++t) {
                        int variant = 0;
                        switch (pick[t].type) {
                            case BuyType::kBuyInfantry: variant = 0; break;
                            case BuyType::kConvertToSpecial: variant = 1 + pick[t].unit_type; break;
                            case BuyType::kBuildOutpost: variant = 3 + pick[t].unit_type; break;
                            default: variant = 6 + pick[t].upgrade; break;
                        }
                        if (variant > 8) variant = 8;
                        const int cell = pick[t].hex * 9 + variant;
                        if (std::find(seen.begin(), seen.end(), cell) != seen.end()) continue;
                        seen.push_back(cell);
                        reduced.push_back(pick[t]);
                    }
                    const int base_n = reduced.size();
                    for (int pass = 0; pass < kBuyRepeatPasses && base_n > 0; ++pass) {
                        bool appended = false;
                        for (int t = 0; t < base_n; ++t) {
                            const BuyAction& act = reduced[t];
                            if (act.type != BuyType::kBuyInfantry &&
                                act.type != BuyType::kConvertToSpecial) {
                                continue;
                            }
                            if (reduced.size() >= kMaxChosenBuy) break;
                            reduced.push_back(act);
                            appended = true;
                        }
                        if (!appended) break;
                    }
                    *dst_buy = reduced;
                }
            }
        }
    }
}

}  // namespace

namespace {
struct LookaheadReport {
    ~LookaheadReport() {
        const long long d = g_la_decisions.load();
        if (d == 0) return;
        const long long c = g_la_changed.load();
        std::fprintf(stderr,
                     "buy lookahead: %lld decisions, changed the answer %lld "
                     "(%.1f%%); of those %lld differ in WHICH cells (%.1f%% of all) "
                     "and %lld differ only in QUANTITY (%.1f%% of all, invisible to "
                     "the training label)\n",
                     d, c, 100.0 * double(c) / double(d), g_la_cells_differ.load(),
                     100.0 * double(g_la_cells_differ.load()) / double(d),
                     g_la_qty_only.load(),
                     100.0 * double(g_la_qty_only.load()) / double(d));
    }
};
LookaheadReport g_la_report;
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
