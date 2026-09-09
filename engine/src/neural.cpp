// The neural policy, in LibTorch (PLAN.md §10, M9c).
//
// Only compiled when OO_LIBTORCH_DIR is set; see neural_stub.cpp for the other
// build. Nothing else in the engine includes a torch header.

#include "oo/neural.hpp"

#include "oo/buy.hpp"
#include "oo/encode.hpp"
#include "oo/grid.hpp"

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

// --- wire layout ------------------------------------------------------------
// Byte offsets into EncodedPosition. Kept as constants next to the tensor code
// that slices them, because a mismatch here produces a model that trains on
// garbage without ever erroring.
constexpr int kHeaderOff = 0;
constexpr int kFactionOff = static_cast<int>(sizeof(HeaderRecord));
constexpr int kHexOff = kFactionOff + static_cast<int>(sizeof(FactionRecord)) * MAX_FACTIONS;
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

// --- model ------------------------------------------------------------------

struct BlockImpl : torch::nn::Module {
    BlockImpl(int d, int heads, int ffn_mult)
        : n_heads(heads),
          head_dim(d / heads),
          norm1(torch::nn::LayerNormOptions({d})),
          norm2(torch::nn::LayerNormOptions({d})),
          qkv(torch::nn::LinearOptions(d, 3 * d)),
          proj(torch::nn::LinearOptions(d, d)),
          fc1(torch::nn::LinearOptions(d, d * ffn_mult)),
          fc2(torch::nn::LinearOptions(d * ffn_mult, d)) {
        register_module("norm1", norm1);
        register_module("norm2", norm2);
        register_module("qkv", qkv);
        register_module("proj", proj);
        register_module("fc1", fc1);
        register_module("fc2", fc2);
    }

    // `bias` is [heads, T, T], already gathered from the bucket table.
    torch::Tensor forward(torch::Tensor x, const torch::Tensor& bias) {
        const auto B = x.size(0);
        const auto T = x.size(1);

        // Pre-LN: normalise going in, add the residual raw. More stable to train
        // than post-LN and the standard choice for a net this shallow.
        auto h = norm1->forward(x);
        auto q_k_v = qkv->forward(h).view({B, T, 3, n_heads, head_dim}).permute({2, 0, 3, 1, 4});
        auto q = q_k_v[0], k = q_k_v[1], v = q_k_v[2];
        // attn_mask as a FLOAT tensor is ADDED to the scores, which is exactly
        // what a relative position bias is. (A bool mask would instead mean
        // allowed/blocked, so passing the bias as one would silently threshold
        // it.) This is the ATen op; there is no nn::functional wrapper.
        auto attn = torch::scaled_dot_product_attention(q, k, v, bias.unsqueeze(0));
        attn = attn.transpose(1, 2).reshape({B, T, n_heads * head_dim});
        x = x + proj->forward(attn);

        h = norm2->forward(x);
        x = x + fc2->forward(F::gelu(fc1->forward(h)));
        return x;
    }

    int n_heads, head_dim;
    torch::nn::LayerNorm norm1, norm2;
    torch::nn::Linear qkv, proj, fc1, fc2;
};
TORCH_MODULE(Block);

struct NetImpl : torch::nn::Module {
    NetImpl(const NeuralConfig& cfg, int n_hex)
        : d(cfg.d_model),
          nf(cfg.num_factions),
          hexes(n_hex),
          hex_in(torch::nn::LinearOptions(kHexFeatures, cfg.d_model)),
          faction_in(torch::nn::LinearOptions(kFactionFeatures, cfg.d_model)),
          header_in(torch::nn::LinearOptions(kHeaderFeatures, cfg.d_model)),
          slot_emb(torch::nn::EmbeddingOptions(MAX_FACTIONS, cfg.d_model)),
          head_move(torch::nn::LinearOptions(cfg.d_model, NUM_DIRECTIONS)),
          head_pass(torch::nn::LinearOptions(cfg.d_model, 1)),
          head_hex(torch::nn::LinearOptions(cfg.d_model, 1)),
          head_buy(torch::nn::LinearOptions(cfg.d_model, 9)),
          head_target(torch::nn::LinearOptions(cfg.d_model, 1)),
          head_resource(torch::nn::LinearOptions(cfg.d_model, 2)),
          head_swap(torch::nn::LinearOptions(cfg.d_model, 2)),
          head_rect(torch::nn::LinearOptions(cfg.d_model, MAX_BATTLE_CONTRIB * NUM_UNIT_TYPES)),
          head_value(torch::nn::LinearOptions(cfg.d_model, 1)) {
        register_module("hex_in", hex_in);
        register_module("faction_in", faction_in);
        register_module("header_in", header_in);
        register_module("slot_emb", slot_emb);
        for (int i = 0; i < cfg.stem_layers; ++i) {
            // The hex-conv stem: a 7-tap gather over the hex's own value and its
            // six neighbours (§10.6). Cheap, and it means attention does not have
            // to rediscover adjacency - which is what the movement action space
            // is defined over.
            auto layer = torch::nn::Linear(torch::nn::LinearOptions(cfg.d_model * 7, cfg.d_model));
            register_module("stem" + std::to_string(i), layer);
            stem.push_back(layer);
        }
        for (int i = 0; i < cfg.n_blocks; ++i) {
            auto b = Block(cfg.d_model, cfg.n_heads, cfg.ffn_mult);
            register_module("block" + std::to_string(i), b);
            blocks.push_back(b);
        }
        bias_table = register_parameter("bias_table",
                                        torch::zeros({cfg.n_heads, kNumBuckets}));
        register_module("head_move", head_move);
        register_module("head_pass", head_pass);
        register_module("head_hex", head_hex);
        register_module("head_buy", head_buy);
        register_module("head_target", head_target);
        register_module("head_resource", head_resource);
        register_module("head_swap", head_swap);
        register_module("head_rect", head_rect);
        register_module("head_value", head_value);
    }

    // raw: [B, kPosBytes] uint8. Returns the token stream [B, T, d].
    torch::Tensor embed(const torch::Tensor& raw) {
        const auto B = raw.size(0);
        auto i32 = raw.to(torch::kInt32);

        // --- hex tokens ------------------------------------------------------
        auto hx = i32.slice(1, kHexOff, kHexOff + 12 * MAX_HEXES)
                      .view({B, MAX_HEXES, 12})
                      .slice(1, 0, hexes);
        auto terrain = hx.select(2, 0);
        auto city = hx.select(2, 1);
        auto city_flags = hx.select(2, 2);
        auto army = hx.select(2, 3);
        auto units = hx.slice(2, 4, 7).to(torch::kFloat32);
        auto flags = hx.select(2, 7);
        auto battle_round = hx.select(2, 8).to(torch::kFloat32);
        auto battle_factions = hx.select(2, 10) + hx.select(2, 11) * 256;

        std::vector<torch::Tensor> hf;
        hf.push_back(F::one_hot(terrain.to(torch::kLong), NUM_TERRAIN_TYPES).to(torch::kFloat32));
        for (int k = 0; k < 7; ++k) hf.push_back(bit(flags, k).to(torch::kFloat32).unsqueeze(-1));
        hf.push_back(F::one_hot(city.to(torch::kLong), MAX_FACTIONS + 1).to(torch::kFloat32));
        hf.push_back(bit(city_flags, 0).to(torch::kFloat32).unsqueeze(-1));
        hf.push_back(F::one_hot(torch::bitwise_and(torch::bitwise_right_shift(city_flags, 1), 3).to(torch::kLong), 4)
                         .to(torch::kFloat32));
        hf.push_back(F::one_hot(army.to(torch::kLong), MAX_FACTIONS + 1).to(torch::kFloat32));
        hf.push_back(units / double(MAX_STACK_SIZE));
        hf.push_back((units.sum(-1) / double(MAX_STACK_SIZE)).unsqueeze(-1));
        hf.push_back((battle_round / 5.0).unsqueeze(-1));
        for (int k = 0; k < MAX_FACTIONS; ++k) {
            hf.push_back(bit(battle_factions, k).to(torch::kFloat32).unsqueeze(-1));
        }
        auto hex_feat = torch::cat(hf, -1);
        TORCH_CHECK(hex_feat.size(-1) == kHexFeatures, "hex feature width drifted: got ",
                    hex_feat.size(-1), " expected ", kHexFeatures);
        auto hex_tok = hex_in->forward(hex_feat.to(dtype()));

        // --- hex-conv stem ---------------------------------------------------
        for (auto& layer : stem) {
            // neighbours_idx is [hexes, 7]: self, then the six directions, with
            // off-board directions pointing at a zero row appended at the end.
            auto padded = torch::cat({hex_tok, torch::zeros({B, 1, d}, hex_tok.options())}, 1);
            auto gathered = padded.index_select(1, neighbours_idx.view({-1}))
                                .view({B, hexes, 7 * d});
            hex_tok = hex_tok + F::gelu(layer->forward(gathered));
        }

        // --- faction tokens ---------------------------------------------------
        auto fc = i32.slice(1, kFactionOff, kFactionOff + 24 * MAX_FACTIONS)
                      .view({B, MAX_FACTIONS, 24})
                      .slice(1, 0, nf);
        std::vector<torch::Tensor> ff;
        ff.push_back(fc.select(2, 0).to(torch::kFloat32).unsqueeze(-1));  // active
        ff.push_back(fc.select(2, 1).to(torch::kFloat32).unsqueeze(-1));  // is_me
        ff.push_back((fc.select(2, 2).to(torch::kFloat32) / double(kOutpostCap)).unsqueeze(-1));
        ff.push_back(fc.slice(2, 3, 6).to(torch::kFloat32) / 20.0);       // units by type
        ff.push_back((fc.select(2, 6).to(torch::kFloat32) / 10.0).unsqueeze(-1));  // settle order
        ff.push_back((read_i16(fc, 8).to(torch::kFloat32) / 50.0).unsqueeze(-1));  // gold
        std::vector<torch::Tensor> res;
        for (int r = 0; r < NUM_RESOURCES; ++r) res.push_back(read_i16(fc, 10 + 2 * r).unsqueeze(-1));
        ff.push_back(torch::cat(res, -1).to(torch::kFloat32) / 20.0);
        ff.push_back((read_i16(fc, 18).to(torch::kFloat32) / double(kVpToWin)).unsqueeze(-1));
        ff.push_back((read_i16(fc, 20).to(torch::kFloat32) / 20.0).unsqueeze(-1));
        ff.push_back((read_i16(fc, 22).to(torch::kFloat32) / double(kVpToWin)).unsqueeze(-1));
        auto fac_feat = torch::cat(ff, -1);
        TORCH_CHECK(fac_feat.size(-1) == kFactionFeatures, "faction feature width drifted: got ",
                    fac_feat.size(-1), " expected ", kFactionFeatures);
        // Slot identity comes from a learned embedding rather than a one-hot,
        // since the slot IS the relative faction index and the net should be free
        // to give "me" its own vector.
        auto fac_tok = faction_in->forward(fac_feat.to(dtype())) +
                       slot_emb->forward(torch::arange(nf, torch::TensorOptions()
                                                               .dtype(torch::kLong)
                                                               .device(raw.device())))
                           .unsqueeze(0);

        // --- CLS --------------------------------------------------------------
        auto hd = i32.slice(1, kHeaderOff, kHeaderOff + 16);
        std::vector<torch::Tensor> gf;
        gf.push_back((hd.select(1, 1).to(torch::kFloat32) / 8.0).unsqueeze(-1));   // radius
        gf.push_back((hd.select(1, 2).to(torch::kFloat32) / 10.0).unsqueeze(-1));  // num_factions
        gf.push_back(F::one_hot(hd.select(1, 3).to(torch::kLong), kNumDecisionKinds).to(torch::kFloat32));
        gf.push_back((hd.select(1, 4).to(torch::kFloat32) / 3.0).unsqueeze(-1));   // step
        gf.push_back((read_i16(hd, 6).to(torch::kFloat32) / 331.0).unsqueeze(-1));  // num_hexes
        gf.push_back((read_i16(hd, 8).to(torch::kFloat32) / 50.0).unsqueeze(-1));   // turn
        gf.push_back((read_i16(hd, 12).to(torch::kFloat32) / double(kVpToWin)).unsqueeze(-1));
        auto head_feat = torch::cat(gf, -1);
        TORCH_CHECK(head_feat.size(-1) == kHeaderFeatures, "header feature width drifted: got ",
                    head_feat.size(-1), " expected ", kHeaderFeatures);
        auto cls_tok = header_in->forward(head_feat.to(dtype())).unsqueeze(1);

        return torch::cat({hex_tok, fac_tok, cls_tok}, 1);
    }

    // Returns the trunk output [B, T, d].
    torch::Tensor trunk(const torch::Tensor& raw) {
        auto x = embed(raw);
        auto bias = bias_table.index_select(1, bucket_idx.view({-1}))
                        .view({bias_table.size(0), bucket_idx.size(0), bucket_idx.size(1)});
        for (auto& b : blocks) x = b->forward(x, bias);
        return x;
    }

    // Whatever the weights were converted to. The byte expansion above works in
    // int32/float32 regardless, so every projection casts on the way in.
    torch::ScalarType dtype() const { return hex_in->weight.scalar_type(); }

    int d, nf, hexes;
    torch::nn::Linear hex_in, faction_in, header_in;
    torch::nn::Embedding slot_emb;
    std::vector<torch::nn::Linear> stem;
    std::vector<Block> blocks;
    torch::Tensor bias_table;
    torch::Tensor neighbours_idx;  // [hexes, 7] long
    torch::Tensor bucket_idx;      // [T, T] long

    torch::nn::Linear head_move, head_pass, head_hex, head_buy, head_target, head_resource,
        head_swap, head_rect, head_value;
};
TORCH_MODULE(Net);

}  // namespace

// --- the policy -------------------------------------------------------------

namespace {

class TorchPolicy : public BatchedPolicy {
public:
    TorchPolicy(const NeuralConfig& cfg, const HexGrid& grid)
        : cfg_(cfg),
          grid_(grid),
          hexes_(grid.num_hexes()),
          tokens_(grid.num_hexes() + cfg.num_factions + 1),
          device_(cfg.cuda ? torch::kCUDA : torch::kCPU) {
        torch::manual_seed(cfg.seed);
        net_ = Net(cfg, hexes_);

        // Neighbour gather table: column 0 is self, 1..6 the directions. An
        // off-board direction points at row `hexes_`, a zero row appended at
        // gather time, so the stem needs no masking.
        std::vector<int64_t> nb(static_cast<size_t>(hexes_) * 7);
        for (int h = 0; h < hexes_; ++h) {
            nb[static_cast<size_t>(h) * 7] = h;
            for (int dir = 0; dir < NUM_DIRECTIONS; ++dir) {
                const int j = grid.neighbour(h, dir);
                nb[static_cast<size_t>(h) * 7 + 1 + dir] = j >= 0 ? j : hexes_;
            }
        }
        net_->neighbours_idx =
            torch::from_blob(nb.data(), {hexes_, 7}, torch::kLong).clone().to(device_);

        // Distance-bias bucket table over the whole token stream.
        std::vector<int64_t> bk(static_cast<size_t>(tokens_) * static_cast<size_t>(tokens_));
        const int fac0 = hexes_;
        const int cls = tokens_ - 1;
        for (int i = 0; i < tokens_; ++i) {
            for (int j = 0; j < tokens_; ++j) {
                int b;
                const bool ih = i < fac0, jh = j < fac0;
                const bool ic = i == cls, jc = j == cls;
                if (ih && jh) {
                    b = distance_bucket(grid.distance(i, j));
                } else if (ic && jc) {
                    b = kBucketClsCls;
                } else if (ic || jc) {
                    b = (ih || jh) ? kBucketHexCls : kBucketFactionCls;
                } else if (ih || jh) {
                    b = kBucketHexFaction;
                } else {
                    b = kBucketFactionFaction;
                }
                bk[static_cast<size_t>(i) * static_cast<size_t>(tokens_) + static_cast<size_t>(j)] =
                    b;
            }
        }
        net_->bucket_idx =
            torch::from_blob(bk.data(), {tokens_, tokens_}, torch::kLong).clone().to(device_);

        net_->to(device_);
        if (cfg.fp16 && cfg.cuda) net_->to(torch::kFloat16);
        net_->eval();

        params_ = 0;
        for (const auto& p : net_->parameters()) params_ += p.numel();
    }

    void evaluate(const DecisionRequest* requests, DecisionAnswer* answers, int n) override;
    void evaluate_impl(const DecisionRequest* requests, DecisionAnswer* answers, int n);

    std::string describe() const {
        std::ostringstream os;
        os << "d=" << cfg_.d_model << " x" << cfg_.n_blocks << " blocks, " << cfg_.n_heads
           << " heads, " << tokens_ << " tokens, " << (params_ / 1000) << "k params, "
           << (cfg_.cuda ? "cuda" : "cpu") << (cfg_.fp16 && cfg_.cuda ? "/fp16" : "/fp32");
        return os.str();
    }

private:
    NeuralConfig cfg_;
    const HexGrid& grid_;
    int hexes_, tokens_;
    torch::Device device_;
    Net net_{nullptr};
    long long params_ = 0;
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

    auto x = net_->trunk(raw);
    auto hex_tok = x.slice(1, 0, hexes_);
    auto fac_tok = x.slice(1, hexes_, hexes_ + cfg_.num_factions);
    auto cls_tok = x.select(1, tokens_ - 1);

    // Every head runs for the whole batch. They are a rounding error next to the
    // trunk (~0.7 MFLOP against ~370), and computing them all keeps the GPU path
    // branch-free - the CPU then reads only what each request asked for.
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
    auto query_idx = torch::from_blob(query.data(), {n, 1}, torch::kLong).to(device_);
    // [B, 1, d] -> the trunk row for each position's own query hex.
    auto query_tok =
        hex_tok.gather(1, query_idx.unsqueeze(-1).expand({n, 1, hex_tok.size(2)})).squeeze(1);

    // ONE download, not eight. Each `.to(kCPU)` is a full device synchronisation,
    // so eight of them serialised the batch eight times over. Concatenating on
    // the device and copying once costs a single sync.
    //
    // Widths, in order: move 6, pass 1, hex 1, buy 9, target nf, resource 2,
    // swap 2, rect 48. Per-hex heads are flattened so everything is [B, W].
    const int nf = cfg_.num_factions;
    auto flat = torch::cat(
        {
            net_->head_move->forward(hex_tok).reshape({n, hexes_ * NUM_DIRECTIONS}),
            net_->head_pass->forward(cls_tok),
            net_->head_hex->forward(hex_tok).reshape({n, hexes_}),
            net_->head_buy->forward(hex_tok).reshape({n, hexes_ * 9}),
            net_->head_target->forward(fac_tok).reshape({n, nf}),
            net_->head_resource->forward(query_tok),
            net_->head_swap->forward(cls_tok),
            net_->head_rect->forward(query_tok),
        },
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

long long neural_param_count(const NeuralConfig& cfg) {
    const long long d = cfg.d_model;
    long long p = 0;
    p += (kHexFeatures + 1) * d + (kFactionFeatures + 1) * d + (kHeaderFeatures + 1) * d;
    p += MAX_FACTIONS * d;                       // slot embedding
    p += cfg.stem_layers * (7 * d * d + d);      // stem
    p += cfg.n_blocks * (4 * d + 3 * d * d + 3 * d + d * d + d +
                         2 * cfg.ffn_mult * d * d + cfg.ffn_mult * d + d);
    p += cfg.n_heads * kNumBuckets;
    p += (d + 1) * (NUM_DIRECTIONS + 1 + 1 + 9 + 1 + 2 + 2 +
                    MAX_BATTLE_CONTRIB * NUM_UNIT_TYPES + 1);
    return p;
}

std::unique_ptr<BatchedPolicy> make_neural_policy(const NeuralConfig& cfg, std::string& error) {
    if (cfg.cuda && !torch::cuda::is_available()) {
        error = "CUDA requested but not available to LibTorch";
        return nullptr;
    }
    if (cfg.radius < 1 || cfg.radius > 8) {
        error = "radius must be 1-8";
        return nullptr;
    }
    if (cfg.d_model % cfg.n_heads != 0) {
        error = "d_model must be divisible by n_heads";
        return nullptr;
    }
    return std::make_unique<TorchPolicy>(cfg, HexGrid::shared(cfg.radius));
}

std::string neural_describe(const BatchedPolicy& policy) {
    const auto* p = dynamic_cast<const TorchPolicy*>(&policy);
    return p ? p->describe() : std::string("(not a neural policy)");
}

}  // namespace oo
