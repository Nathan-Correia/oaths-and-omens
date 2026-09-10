// Head-to-head evaluation (PLAN.md §12.1, M10a).
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
//   among IDENTICAL agents in an earlier 3-tactician/5-greedy run (§6.8a), so an
//   unmirrored comparison largely measures placement order.
//
//   A WILSON INTERVAL rather than the normal approximation, which misbehaves
//   near 0 and 1 - exactly where a strong agent lands against a weak one.

#include "oo/agent.hpp"
#include "oo/batch.hpp"
#include "oo/neural.hpp"
#include "oo/run.hpp"

#include <cmath>
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
                 "  --b-weights PATH  ditto for side B\n"
                 "  --games N         seed pairs; each is played twice (default 500)\n"
                 "  --ladder          play A against random, greedy, marshal, tactician\n"
                 "  --radius N        board radius (default 7)\n"
                 "  --factions N      factions, split evenly between sides (default 8)\n"
                 "  --seed N          base seed (default 1)\n"
                 "  --max-turns N     safety cap (default 100)\n"
                 "  --in-flight N     games in flight when a side is neural (default 256)\n"
                 "  --threads N       threads when both sides are scripted\n");
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

struct Side {
    std::string name;
    std::string weights;
    bool neural = false;
    AgentKind kind = AgentKind::kRandom;
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

        BatchedPolicy* pa = a.neural ? net_a.get() : &scripted;
        BatchedPolicy* pb = b.neural ? net_b.get() : &scripted;
        SplitPolicy split(pa, pb, a_seats, nf);

        std::vector<GameResult> results(static_cast<size_t>(pairs));
        const bool any_neural = a.neural || b.neural;
        run_games_batched(cfg, specs.data(), results.data(), pairs,
                          any_neural ? in_flight : threads, any_neural ? in_flight : 1, split);

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
    int pairs = 500, radius = 7, factions = 8, max_turns = 100, in_flight = 256, threads = 0;
    long long base_seed = 1;
    bool ladder = false;

    for (int i = 1; i < argc; ++i) {
        const std::string s = argv[i];
        auto next = [&]() -> std::string { return (i + 1 < argc) ? argv[++i] : ""; };
        if (s == "--a") a_name = next();
        else if (s == "--b") b_name = next();
        else if (s == "--a-weights") a_weights = next();
        else if (s == "--b-weights") b_weights = next();
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
        bool ok = true;
        const Result r = play_matchup(a, b, cfg, base_seed, pairs, in_flight, threads, ok);
        if (!ok) return 1;
        report(a.name, b.name, r);
    }
    return 0;
}
