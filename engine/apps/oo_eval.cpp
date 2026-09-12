// Head-to-head evaluation (PLAN.md Â§12.1, M10a).
//
//   oo_eval --a tactician --b marshal --games 2000
//   oo_eval --a neural --a-weights checkpoints/bc.ts --b tactician --games 2000
//   oo_eval --a neural --a-weights checkpoints/bc.ts --ladder
//
// Answers "is A stronger than B?" with an interval, which `oo_run`'s summary
// cannot: a win rate over a few hundred games is mostly noise, and telling a
// 52 % agent from a 50 % one needs on the order of 10 000.
//
// Three things make the number mean something:
//
//   PAIRED SEEDS. Every seed is played TWICE with the sides swapped, so the same
//   board and the same dice are seen from both sides. Terrain luck cancels
//   instead of adding variance.
//
//   SEAT ROTATION comes free with that mirror: seat win rates ranged 9-19 %
//   among IDENTICAL agents in an earlier 3-tactician/5-greedy run (Â§6.8a), so an
//   unmirrored comparison largely measures placement order.
//
//   A WILSON INTERVAL rather than the normal approximation, which misbehaves
//   near 0 and 1 - exactly where a strong agent lands against a weak one.

#include "oo/agent.hpp"
#include "oo/batch.hpp"
#include "oo/neural.hpp"
#include "oo/run.hpp"

#include <cmath>
#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

// Everything here is engine-side; qualifying every name would only add noise.
using namespace oo;

namespace {

void usage() {
    std::fprintf(stderr,
                 "usage: oo_eval [options]\n"
                 "  --a NAME          side A: an agent kind, or 'neural'\n"
                 "  --b NAME          side B (ignored with --ladder)\n"
                 "  --a-weights PATH  TorchScript for side A when it is neural\n"
                 "  --a-lookahead K   one-ply value lookahead on buy for side A:\n"
                 "                    score K candidate purchase sets with\n"
                 "                    the value head, keep the best (0/1 off)\n"
                 "  --b-lookahead K   the same for side B\n"
                 "  --lookahead-tau T Gumbel noise generating the candidates\n"
                 "  --b-weights PATH  ditto for side B\n"
                 "  --games N         seed pairs; each is played twice (default 500)\n"
                 "  --ladder          play A against random, greedy, marshal, tactician\n"
                 "  --radius N        board radius (default 7)\n"
                 "  --factions N      factions, split evenly between sides (default 8)\n"
                 "  --seed N          base seed (default 1)\n"
                 "  --max-turns N     safety cap (default 100)\n"
                 "  --in-flight N     games in flight when a side is neural (default 256)\n"
                 "  --threads N       threads when both sides are scripted\n"
                 "  --a-fallback K,K  ABLATION: side A uses a scripted agent for these\n"
                 "                    decision kinds (buy, movement, cavalry, target,\n"
                 "                    rectification, resource, placement, draft, swap)\n"
                 "  --a-fallback-agent NAME   which agent that is (default tactician)\n");
}

std::vector<std::string> split_commas(const std::string& in) {
    std::vector<std::string> out;
    size_t start = 0;
    while (start <= in.size()) {
        const size_t c = in.find(',', start);
        const size_t end = c == std::string::npos ? in.size() : c;
        if (end > start) out.push_back(in.substr(start, end - start));
        if (c == std::string::npos) break;
        start = c + 1;
    }
    return out;
}

// Wilson score interval. Correct near 0 and 1, where the textbook
// p +- z*sqrt(p(1-p)/n) can produce bounds outside [0, 1] and badly understate
// uncertainty on a lopsided matchup.
struct Interval {
    double lo, hi, mid;
};

Interval wilson(long long wins, long long n, double z = 1.96) {
    if (n == 0) return {0.0, 1.0, 0.5};
    const double p = double(wins) / double(n);
    const double z2 = z * z;
    const double denom = 1.0 + z2 / n;
    const double centre = (p + z2 / (2 * n)) / denom;
    const double half = z * std::sqrt(p * (1 - p) / n + z2 / (4.0 * n * n)) / denom;
    return {centre - half, centre + half, centre};
}

// Routes each decision to whichever side owns that seat. This is what lets one
// game contain both a network and a scripted agent - without it a "head to head"
// would have to be two separate runs, which is not the same experiment.
class SplitPolicy : public BatchedPolicy {
public:
    SplitPolicy(BatchedPolicy* a, BatchedPolicy* b, const bool* a_seats, int num_factions)
        : a_(a), b_(b) {
        for (int f = 0; f < MAX_FACTIONS; ++f) a_seats_[f] = f < num_factions && a_seats[f];
    }

    void on_game_start(const GameState* s, int i) override {
        a_->on_game_start(s, i);
        if (b_ != a_) b_->on_game_start(s, i);
    }
    void on_game_end(const GameState* s) override {
        a_->on_game_end(s);
        if (b_ != a_) b_->on_game_end(s);
    }

    void evaluate(const DecisionRequest* reqs, DecisionAnswer* ans, int n) override {
        // Split the batch, evaluate each side once, scatter the answers back.
        // Evaluating per request would defeat the entire point of batching.
        std::vector<DecisionRequest> ra, rb;
        std::vector<DecisionAnswer> aa, ab;
        std::vector<int> ia, ib;
        for (int i = 0; i < n; ++i) {
            const int seat = reqs[i].kind == DecisionKind::kRectification ? reqs[i].winner
                                                                         : reqs[i].faction;
            const bool mine = seat >= 0 && a_seats_[seat];
            if (mine) {
                ra.push_back(reqs[i]);
                aa.push_back(ans[i]);
                ia.push_back(i);
            } else {
                rb.push_back(reqs[i]);
                ab.push_back(ans[i]);
                ib.push_back(i);
            }
        }
        if (!ra.empty()) a_->evaluate(ra.data(), aa.data(), int(ra.size()));
        if (!rb.empty()) b_->evaluate(rb.data(), ab.data(), int(rb.size()));
        for (size_t k = 0; k < ia.size(); ++k) ans[ia[k]] = aa[k];
        for (size_t k = 0; k < ib.size(); ++k) ans[ib[k]] = ab[k];
    }

private:
    BatchedPolicy* a_;
    BatchedPolicy* b_;
    bool a_seats_[MAX_FACTIONS] = {};
};

// Routes by decision KIND rather than by seat: side A plays the network for
// every kind EXCEPT the listed ones, which fall back to a scripted agent.
//
// This is an ablation, not a way to play. It answers "if this head were
// perfect, how much strength would we gain?", which is the only way to know
// which head is actually costing games - agreement percentages cannot say,
// because a head can disagree constantly and cost nothing (the teacher's buy
// targeting is a coin flip) or agree mostly and cost a lot.
class KindFallbackPolicy : public BatchedPolicy {
public:
    KindFallbackPolicy(BatchedPolicy* net, BatchedPolicy* scripted, const bool* fallback)
        : net_(net), scripted_(scripted) {
        for (int k = 0; k < kNumDecisionKinds; ++k) fallback_[k] = fallback[k];
    }

    void on_game_start(const GameState* s, int i) override {
        net_->on_game_start(s, i);
        scripted_->on_game_start(s, i);
    }
    void on_game_end(const GameState* s) override {
        net_->on_game_end(s);
        scripted_->on_game_end(s);
    }

    void evaluate(const DecisionRequest* reqs, DecisionAnswer* ans, int n) override {
        std::vector<DecisionRequest> rn, rs;
        std::vector<DecisionAnswer> an, as;
        std::vector<int> in, is;
        for (int i = 0; i < n; ++i) {
            if (fallback_[static_cast<int>(reqs[i].kind)]) {
                rs.push_back(reqs[i]); as.push_back(ans[i]); is.push_back(i);
            } else {
                rn.push_back(reqs[i]); an.push_back(ans[i]); in.push_back(i);
            }
        }
        if (!rn.empty()) net_->evaluate(rn.data(), an.data(), int(rn.size()));
        if (!rs.empty()) scripted_->evaluate(rs.data(), as.data(), int(rs.size()));
        for (size_t k = 0; k < in.size(); ++k) ans[in[k]] = an[k];
        for (size_t k = 0; k < is.size(); ++k) ans[is[k]] = as[k];
    }

private:
    BatchedPolicy* net_;
    BatchedPolicy* scripted_;
    bool fallback_[kNumDecisionKinds] = {};
};

struct Side {
    std::string name;
    std::string weights;
    bool neural = false;
    int lookahead = 0;   // one-ply value lookahead on buy; 0/1 disables
    float lookahead_tau = 1.0f;
    bool lookahead_cells_only = false;
    // Squeeze this side's buy answers through the neural buy representation.
    bool buy_roundtrip = false;
    // Ask this side's buy question twice and report self-agreement.
    bool buy_selfagree = false;
    float buy_threshold = 0.0f;
    float buy_temperature = 0.0f;
    AgentKind kind = AgentKind::kRandom;
    // Ablation: decision kinds this side hands back to a scripted agent.
    bool fallback[kNumDecisionKinds] = {};
    bool any_fallback = false;
    AgentKind fallback_kind = AgentKind::kTactician;
};

bool parse_side(const std::string& name, const std::string& weights, Side& out) {
    out.name = name;
    out.weights = weights;
    if (name == "neural") {
        out.neural = true;
        if (weights.empty()) {
            std::fprintf(stderr, "side '%s' is neural but no weights were given\n", name.c_str());
            return false;
        }
        return true;
    }
    if (!agent_kind_from_name(name.c_str(), out.kind)) {
        std::fprintf(stderr, "unknown agent '%s'\n", name.c_str());
        return false;
    }
    return true;
}

// Asks the wrapped policy the SAME buy question twice and records how often it
// agrees with itself.
//
// This measures the ceiling on imitating it. Tactician's buy runs through
// greedy_buy_common, which does rng.shuffle over the hex list and
// rng.choice_index among equal-scoring infantry actions, so its answer is a
// random variable given the position. A network trained with per-cell BCE
// learns the marginal probability of each cell and decodes the most likely
// set; it cannot agree with a sampled draw more often than two draws agree
// with each other. If that self-agreement is near the 45.2% the network
// already reaches, imitation is finished and no amount of capacity, gradient
// share or data will move it.
//
// The second call is the one that plays, so the games are ordinary tactician
// games - only the RNG stream is consumed twice as fast.
class BuySelfAgreePolicy : public BatchedPolicy {
public:
    explicit BuySelfAgreePolicy(BatchedPolicy* inner) : inner_(inner) {
        for (int k = 0; k < kNumDecisionKinds; ++k) {
            total_[k] = 0;
            pair_num_[k] = 0;
            pair_den_[k] = 0;
            modal_[k] = 0;
        }
    }

    void evaluate(const DecisionRequest* requests, DecisionAnswer* answers, int n) override {
        std::vector<int> buys;
        for (int i = 0; i < n; ++i) buys.push_back(i);
        if (buys.empty()) {
            inner_->evaluate(requests, answers, n);
            return;
        }
        // Draw the same decision kDraws times. The last draw is the one that
        // plays, so these are ordinary games with a faster-consumed RNG.
        //
        // Two statistics come out, and the difference between them is the
        // point. Pairwise agreement estimates sum(p^2) - how often two samples
        // of the teacher collide, which is what a sampling imitator could
        // reach. Modal frequency estimates max(p) - how often the single most
        // likely set comes up, which is what a DETERMINISTIC imitator like our
        // decoder can reach. max(p) >= sum(p^2) always, with equality only for
        // a uniform distribution, so quoting self-agreement alone understates
        // the ceiling.
        std::vector<std::vector<std::vector<int> > > draws(buys.size());
        std::vector<ChosenBuyActions> scratch(buys.size());
        std::vector<ChosenBuyActions*> saved(buys.size());
        for (size_t j = 0; j < buys.size(); ++j) saved[j] = answers[buys[j]].buy;

        for (int d = 0; d < kDraws; ++d) {
            const bool last = (d == kDraws - 1);
            for (size_t j = 0; j < buys.size(); ++j) {
                answers[buys[j]].buy = last ? saved[j] : &scratch[j];
            }
            inner_->evaluate(requests, answers, n);
            for (size_t j = 0; j < buys.size(); ++j) {
                draws[j].push_back(fingerprint(requests[buys[j]], answers[buys[j]]));
            }
        }
        for (size_t j = 0; j < buys.size(); ++j) answers[buys[j]].buy = saved[j];

        for (size_t j = 0; j < buys.size(); ++j) {
            const int kind = static_cast<int>(requests[buys[j]].kind);
            const std::vector<std::vector<int> >& d = draws[j];
            long long pairs = 0, same = 0;
            for (size_t x = 0; x < d.size(); ++x) {
                for (size_t y = x + 1; y < d.size(); ++y) {
                    ++pairs;
                    if (d[x] == d[y]) ++same;
                }
            }
            int best = 0;
            for (size_t x = 0; x < d.size(); ++x) {
                int c = 0;
                for (size_t y = 0; y < d.size(); ++y) {
                    if (d[x] == d[y]) ++c;
                }
                if (c > best) best = c;
            }
            total_[kind].fetch_add(1, std::memory_order_relaxed);
            pair_num_[kind].fetch_add(same, std::memory_order_relaxed);
            pair_den_[kind].fetch_add(pairs, std::memory_order_relaxed);
            modal_[kind].fetch_add(best, std::memory_order_relaxed);
        }
    }
    void on_game_start(const GameState* s, int g) override { inner_->on_game_start(s, g); }
    void on_game_end(const GameState* s) override { inner_->on_game_end(s); }

    void report() const {
        std::printf("  teacher self-agreement, %d draws of each decision:\n"
                    "    %-14s %10s  %9s  %9s\n",
                    kDraws, "kind", "decisions", "pairwise", "modal");
        for (int k = 0; k < kNumDecisionKinds; ++k) {
            const long long t = total_[k].load();
            if (t == 0) continue;
            const long long den = pair_den_[k].load();
            const double pairwise = den ? double(pair_num_[k].load()) / double(den) : 0.0;
            const double modal = double(modal_[k].load()) / double(t * kDraws);
            std::printf("    %-14s %10lld  %8.1f%%  %8.1f%%\n",
                        decision_kind_name(static_cast<DecisionKind>(k)), t,
                        100.0 * pairwise, 100.0 * modal);
        }
        std::printf("    pairwise = sum p^2, the ceiling for a SAMPLING imitator;\n"
                    "    modal    = max p,   the ceiling for a DETERMINISTIC one\n");
    }

private:
    // A comparable fingerprint of an answer. Only the fields that kind uses.
    static std::vector<int> fingerprint(const DecisionRequest& r, const DecisionAnswer& a) {
        switch (r.kind) {
            case DecisionKind::kBuy:
                return a.buy ? cells(*a.buy) : std::vector<int>();
            case DecisionKind::kMovement:
            case DecisionKind::kCavalry:
                return a.flag ? std::vector<int>{1, a.move.hex, a.move.dir}
                              : std::vector<int>{0};
            case DecisionKind::kTarget:
            case DecisionKind::kPlacement:
            case DecisionKind::kDraft:
                return std::vector<int>{a.index};
            case DecisionKind::kResource:
                return std::vector<int>{static_cast<int>(a.resource)};
            case DecisionKind::kSwap:
                return std::vector<int>{a.flag ? 1 : 0};
            default: {  // rectification
                std::vector<int> v;
                if (a.send_back) {
                    for (int i = 0; i < a.send_back->size(); ++i) {
                        const SendBackEntry& e = (*a.send_back)[i];
                        v.push_back(e.origin_hex);
                        for (int t = 0; t < NUM_UNIT_TYPES; ++t) v.push_back(e.units[t]);
                    }
                }
                return v;
            }
        }
    }

    static std::vector<int> cells(const ChosenBuyActions& b) {
        std::vector<int> v;
        for (int t = 0; t < b.size(); ++t) {
            int q = 0;
            switch (b[t].type) {
                case BuyType::kBuyInfantry: q = 0; break;
                case BuyType::kConvertToSpecial: q = 1 + b[t].unit_type; break;
                case BuyType::kBuildOutpost: q = 3 + b[t].unit_type; break;
                default: q = 6 + b[t].upgrade; break;
            }
            if (q > 8) q = 8;
            v.push_back(b[t].hex * 9 + q);
        }
        std::sort(v.begin(), v.end());
        v.erase(std::unique(v.begin(), v.end()), v.end());
        return v;
    }

    static const int kDraws = 8;
    BatchedPolicy* inner_;
    mutable std::atomic<long long> total_[kNumDecisionKinds];
    mutable std::atomic<long long> pair_num_[kNumDecisionKinds];
    mutable std::atomic<long long> pair_den_[kNumDecisionKinds];
    mutable std::atomic<long long> modal_[kNumDecisionKinds];
};

// Passes every answer through untouched EXCEPT buy, which is squeezed through
// the neural policy's buy representation and back out again.
//
// That representation is lossy in three specific ways, and this measures what
// the loss is worth in win rate. The training label is a BITMASK over (hex,
// variant) cells, so how many of a repeatable action to buy cannot be
// expressed. The decoder fixes the spend order by TYPE RANK, so preferring
// infantry before an upgrade cannot be expressed. And quantity comes from
// repeat-until-broke, so spending part of the treasury cannot be expressed.
//
// Wrapping TACTICIAN in it answers the question that decides what to build
// next: if tactician survives the round trip, the buy representation is
// adequate and the network simply has not learned to use it; if tactician is
// crippled by it, no amount of training will close the gap and the action
// space itself has to change.
class BuyRoundtripPolicy : public BatchedPolicy {
public:
    explicit BuyRoundtripPolicy(BatchedPolicy* inner) : inner_(inner) {}

    void evaluate(const DecisionRequest* requests, DecisionAnswer* answers, int n) override {
        inner_->evaluate(requests, answers, n);
        for (int i = 0; i < n; ++i) {
            if (requests[i].kind != DecisionKind::kBuy || !answers[i].buy) continue;
            roundtrip(*answers[i].buy);
        }
    }
    void on_game_start(const GameState* s, int g) override { inner_->on_game_start(s, g); }
    void on_game_end(const GameState* s) override { inner_->on_game_end(s); }

private:
    static int variant_of(const BuyAction& a) {
        int v = 0;
        switch (a.type) {
            case BuyType::kBuyInfantry: v = 0; break;
            case BuyType::kConvertToSpecial: v = 1 + a.unit_type; break;
            case BuyType::kBuildOutpost: v = 3 + a.unit_type; break;
            default: v = 6 + a.upgrade; break;
        }
        return v > 8 ? 8 : v;
    }
    static int rank_of(BuyType t) {
        switch (t) {
            case BuyType::kUpgradeOutpost: return 0;
            case BuyType::kBuildOutpost: return 1;
            case BuyType::kBuyInfantry: return 2;
            default: return 3;
        }
    }
    static void roundtrip(ChosenBuyActions& buy) {
        // 1. Collapse to the set of cells a bitmask could hold.
        ChosenBuyActions cells;
        cells.clear();
        std::vector<int> seen;
        for (int t = 0; t < buy.size(); ++t) {
            const int cell = buy[t].hex * 9 + variant_of(buy[t]);
            if (std::find(seen.begin(), seen.end(), cell) != seen.end()) continue;
            seen.push_back(cell);
            cells.push_back(buy[t]);
        }
        // 2. Impose the decoder's spend order.
        std::stable_sort(cells.items, cells.items + cells.count,
                         [](const BuyAction& x, const BuyAction& y) {
                             return rank_of(x.type) < rank_of(y.type);
                         });
        // 3. Re-expand by repeat-until-broke.
        const int base = cells.size();
        for (int pass = 0; pass < 128 && base > 0; ++pass) {
            bool appended = false;
            for (int t = 0; t < base; ++t) {
                const BuyAction& a = cells[t];
                if (a.type != BuyType::kBuyInfantry && a.type != BuyType::kConvertToSpecial) {
                    continue;
                }
                if (cells.size() >= kMaxChosenBuy) break;
                cells.push_back(a);
                appended = true;
            }
            if (!appended) break;
        }
        buy = cells;
    }

    BatchedPolicy* inner_;
};

struct Result {
    long long a_wins = 0, b_wins = 0, draws = 0;
    long long a_vp = 0, b_vp = 0;
    long long turns = 0, games = 0;
};

// Plays `pairs` seeds twice each, sides swapped. Returns A's record.
Result play_matchup(const Side& a, const Side& b, const RunConfig& cfg, long long base_seed,
                    int pairs, int in_flight, int threads, bool& ok) {
    ok = true;
    Result r;
    const int nf = cfg.num_factions;
    const int half = nf / 2;

    // Two orientations of the same seeds: A on the low seats, then A on the high
    // seats. Same board, same dice, sides swapped.
    for (int orientation = 0; orientation < 2; ++orientation) {
        bool a_seats[MAX_FACTIONS] = {};
        for (int f = 0; f < nf; ++f) {
            const bool low = f < half;
            a_seats[f] = (orientation == 0) ? low : !low;
        }

        std::vector<GameSpec> specs(static_cast<size_t>(pairs));
        for (int g = 0; g < pairs; ++g) {
            specs[static_cast<size_t>(g)].seed = base_seed + g;
            for (int f = 0; f < nf; ++f) {
                // Seats a network plays are filled with a placeholder kind; the
                // scripted policy is told to ignore them.
                specs[static_cast<size_t>(g)].seats[f] = a_seats[f] ? a.kind : b.kind;
            }
        }

        ScriptedPolicy scripted(cfg, specs.data(), pairs);
        bool scripted_seats[MAX_FACTIONS] = {};
        for (int f = 0; f < nf; ++f) {
            const bool is_a = a_seats[f];
            scripted_seats[f] = is_a ? !a.neural : !b.neural;
        }
        scripted.set_seats(scripted_seats, nf);

        std::unique_ptr<BatchedPolicy> net_a, net_b;
        std::string err;
        auto make_net = [&](const Side& s) -> std::unique_ptr<BatchedPolicy> {
            NeuralConfig nc;
            nc.weights = s.weights;
            nc.radius = cfg.radius;
            nc.num_factions = cfg.num_factions;
            nc.lookahead_k = s.lookahead;
            nc.lookahead_tau = s.lookahead_tau;
            nc.lookahead_cells_only = s.lookahead_cells_only;
            nc.buy_threshold = s.buy_threshold;
            nc.buy_temperature = s.buy_temperature;
            auto p = make_neural_policy(nc, err);
            if (!p) std::fprintf(stderr, "side %s: %s\n", s.name.c_str(), err.c_str());
            return p;
        };
        if (a.neural) {
            net_a = make_net(a);
            if (!net_a) { ok = false; return r; }
        }
        if (b.neural) {
            net_b = make_net(b);
            if (!net_b) { ok = false; return r; }
        }

        // For an ablation, A's fallback kinds are answered by a scripted agent
        // sitting in A's OWN seats - so it is the same faction, with the same
        // history, just a different brain for those decisions.
        std::vector<GameSpec> fb_specs;
        std::unique_ptr<ScriptedPolicy> fb;
        std::unique_ptr<KindFallbackPolicy> hybrid;
        if (a.neural && a.any_fallback) {
            fb_specs = specs;
            for (int g = 0; g < pairs; ++g) {
                for (int f = 0; f < nf; ++f) {
                    if (a_seats[f]) fb_specs[static_cast<size_t>(g)].seats[f] = a.fallback_kind;
                }
            }
            fb = std::make_unique<ScriptedPolicy>(cfg, fb_specs.data(), pairs);
            fb->set_seats(a_seats, nf);
        }

        BatchedPolicy* pa = a.neural ? net_a.get() : &scripted;
        if (fb) {
            hybrid = std::make_unique<KindFallbackPolicy>(pa, fb.get(), a.fallback);
            pa = hybrid.get();
        }
        std::unique_ptr<BuySelfAgreePolicy> sa;
        if (a.buy_selfagree) {
            sa = std::make_unique<BuySelfAgreePolicy>(pa);
            pa = sa.get();
        }
        std::unique_ptr<BuyRoundtripPolicy> rt;
        if (a.buy_roundtrip) {
            rt = std::make_unique<BuyRoundtripPolicy>(pa);
            pa = rt.get();
        }
        BatchedPolicy* pb = b.neural ? net_b.get() : &scripted;
        SplitPolicy split(pa, pb, a_seats, nf);

        std::vector<GameResult> results(static_cast<size_t>(pairs));
        const bool any_neural = a.neural || b.neural;
        run_games_batched(cfg, specs.data(), results.data(), pairs,
                          any_neural ? in_flight : threads, any_neural ? in_flight : 1, split);

        if (sa) sa->report();

        for (int g = 0; g < pairs; ++g) {
            const GameResult& res = results[static_cast<size_t>(g)];
            ++r.games;
            r.turns += res.turns;
            for (int f = 0; f < nf; ++f) {
                (a_seats[f] ? r.a_vp : r.b_vp) += res.victory_points[f];
            }
            if (res.winner < 0) {
                ++r.draws;
            } else if (a_seats[res.winner]) {
                ++r.a_wins;
            } else {
                ++r.b_wins;
            }
        }
    }
    return r;
}

void report(const std::string& a_name, const std::string& b_name, const Result& r) {
    // Draws are excluded from the rate: with a turn cap, "nobody reached the VP
    // target" says something about the pairing's pace, not about who is better,
    // and folding it in would drag every rate toward 50 %.
    const long long decided = r.a_wins + r.b_wins;
    const Interval w = wilson(r.a_wins, decided);
    std::printf("  %-12s vs %-12s  %5lld-%-5lld (%lld drawn)  win %5.1f%%  [%4.1f, %4.1f]  "
                "avg vp %5.1f v %5.1f  %4.1f turns\n",
                a_name.c_str(), b_name.c_str(), r.a_wins, r.b_wins, r.draws,
                decided ? 100.0 * r.a_wins / decided : 0.0, 100.0 * w.lo, 100.0 * w.hi,
                r.games ? double(r.a_vp) / r.games : 0.0,
                r.games ? double(r.b_vp) / r.games : 0.0,
                r.games ? double(r.turns) / r.games : 0.0);
    if (decided > 0) {
        if (w.lo > 0.5) {
            std::printf("      -> %s is stronger (interval is entirely above 50%%)\n",
                        a_name.c_str());
        } else if (w.hi < 0.5) {
            std::printf("      -> %s is stronger (interval is entirely below 50%%)\n",
                        b_name.c_str());
        } else {
            std::printf("      -> not separated; needs more games\n");
        }
    }
}

}  // namespace

int main(int argc, char** argv) {
    std::string a_name = "tactician", b_name = "marshal", a_weights, b_weights;
    std::string fallback_kinds, fallback_agent = "tactician";
    int pairs = 500, radius = 7, factions = 8, max_turns = 100, in_flight = 256, threads = 0;
    long long base_seed = 1;
    bool ladder = false;
    int a_look = 0, b_look = 0;
    float look_tau = 1.0f;
    bool cells_only = false;
    bool a_roundtrip = false;
    bool a_selfagree = false;
    float a_bt = 0.0f;
    float a_btemp = 0.0f;

    for (int i = 1; i < argc; ++i) {
        const std::string s = argv[i];
        auto next = [&]() -> std::string { return (i + 1 < argc) ? argv[++i] : ""; };
        if (s == "--a") a_name = next();
        else if (s == "--b") b_name = next();
        else if (s == "--a-weights") a_weights = next();
        else if (s == "--b-weights") b_weights = next();
        else if (s == "--a-lookahead") a_look = std::atoi(next().c_str());
        else if (s == "--b-lookahead") b_look = std::atoi(next().c_str());
        else if (s == "--lookahead-tau") look_tau = float(std::atof(next().c_str()));
        else if (s == "--lookahead-cells-only") cells_only = true;
        else if (s == "--a-buy-roundtrip") a_roundtrip = true;
        else if (s == "--a-buy-selfagree") a_selfagree = true;
        else if (s == "--a-buy-threshold") a_bt = float(std::atof(next().c_str()));
        else if (s == "--a-buy-temperature") a_btemp = float(std::atof(next().c_str()));
        else if (s == "--a-fallback") fallback_kinds = next();
        else if (s == "--a-fallback-agent") fallback_agent = next();
        else if (s == "--games") pairs = std::atoi(next().c_str());
        else if (s == "--ladder") ladder = true;
        else if (s == "--radius") radius = std::atoi(next().c_str());
        else if (s == "--factions") factions = std::atoi(next().c_str());
        else if (s == "--seed") base_seed = std::atoll(next().c_str());
        else if (s == "--max-turns") max_turns = std::atoi(next().c_str());
        else if (s == "--in-flight") in_flight = std::atoi(next().c_str());
        else if (s == "--threads") threads = std::atoi(next().c_str());
        else {
            std::fprintf(stderr, "unknown option '%s'\n\n", s.c_str());
            usage();
            return 2;
        }
    }
    if (pairs < 1 || factions < 2 || factions % 2 != 0) {
        std::fprintf(stderr, "--games must be >=1 and --factions an even number >=2\n");
        return 2;
    }
    if (threads <= 0) threads = default_thread_count();

    Side a;
    if (!parse_side(a_name, a_weights, a)) return 2;
    a.lookahead = a_look;
    a.lookahead_tau = look_tau;
    a.lookahead_cells_only = cells_only;
    a.buy_roundtrip = a_roundtrip;
    a.buy_selfagree = a_selfagree;
    a.buy_threshold = a_bt;
    a.buy_temperature = a_btemp;
    if (!fallback_kinds.empty()) {
        for (const std::string& k : split_commas(fallback_kinds)) {
            int found = -1;
            for (int i = 0; i < kNumDecisionKinds; ++i) {
                if (k == decision_kind_name(static_cast<DecisionKind>(i))) found = i;
            }
            if (found < 0) {
                std::fprintf(stderr, "unknown decision kind '%s'\n", k.c_str());
                return 2;
            }
            a.fallback[found] = true;
            a.any_fallback = true;
        }
        if (!agent_kind_from_name(fallback_agent.c_str(), a.fallback_kind)) {
            std::fprintf(stderr, "unknown fallback agent '%s'\n", fallback_agent.c_str());
            return 2;
        }
        std::printf("ABLATION: side A uses %s for [%s]\n", fallback_agent.c_str(),
                    fallback_kinds.c_str());
    }

    RunConfig cfg;
    cfg.radius = radius;
    cfg.num_factions = factions;
    cfg.max_turns = max_turns;

    std::printf("radius %d, %d factions (%d a side), %d seeds x2 orientations = %d games\n",
                radius, factions, factions / 2, pairs, pairs * 2);
    std::printf("paired seeds, sides swapped; 95%% Wilson intervals\n\n");

    std::vector<std::string> opponents;
    if (ladder) {
        // A fixed rung set, so successive checkpoints are comparable to each
        // other and not just to whatever they last played.
        opponents = {"random", "greedy", "marshal", "tactician"};
    } else {
        opponents = {b_name};
    }

    for (const std::string& name : opponents) {
        Side b;
        if (!parse_side(name, name == "neural" ? b_weights : std::string(), b)) return 2;
        b.lookahead = b_look;
        b.lookahead_tau = look_tau;
        b.lookahead_cells_only = cells_only;
        bool ok = true;
        const Result r = play_matchup(a, b, cfg, base_seed, pairs, in_flight, threads, ok);
        if (!ok) return 1;
        report(a.name, b.name, r);
    }
    return 0;
}
