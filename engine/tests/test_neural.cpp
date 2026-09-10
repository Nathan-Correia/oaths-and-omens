// The neural policy runs, answers legally, and is BATCH-INVARIANT (M9c).
//
// Only meaningful when LibTorch is present; the stub build prints "skipping".
//
// Weights are random, so nothing here can say the answers are good. Two things
// it can say, and both are load-bearing:
//
//   1. Every decision kind returns something the engine accepts. An illegal
//      answer shows up as a wrong game, not a crash.
//
//   2. The answer for a position does not depend on which OTHER positions share
//      its batch, or on where in the batch it sits. This is the property the
//      whole §10.9 design rests on - it is why games can block on threads
//      instead of being rewritten as a state machine - and it is precisely what
//      breaks when a per-position gather picks up the wrong row. Every
//      throughput optimisation on this path is a candidate for exactly that bug,
//      so this is the gate they have to pass.
//
// Run on CPU in fp32 so the comparison is exact. On fp16 a different batch size
// can select a different GEMM kernel and shift the last bits, which is a real
// effect (§10.9 accepts it) but would make this test about numerics rather than
// about plumbing.

#include "oo/batch.hpp"
#include "oo/buy.hpp"
#include "oo/movement.hpp"
#include "oo/neural.hpp"
#include "oo/placement.hpp"
#include "oo/run.hpp"
#include "oo/setup.hpp"

#include <algorithm>
#include <array>
#include <cstdio>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

namespace {

int g_failures = 0;

void check(bool ok, const std::string& what) {
    if (!ok) {
        std::cerr << "FAIL: " << what << "\n";
        ++g_failures;
    }
}

// Everything an answer can carry, flattened so two can be compared exactly.
std::string fingerprint(const oo::DecisionAnswer& a, oo::DecisionKind kind) {
    std::string s;
    s += std::to_string(int(kind));
    s += ":" + std::to_string(a.flag ? 1 : 0);
    s += "," + std::to_string(a.move.hex) + "," + std::to_string(int(a.move.dir));
    s += "," + std::to_string(a.index);
    s += "," + std::to_string(int(a.resource));
    if (a.buy) {
        s += ",buy" + std::to_string(a.buy->size());
        for (int i = 0; i < a.buy->size(); ++i) {
            const oo::BuyAction& b = (*a.buy)[i];
            s += "/" + std::to_string(int(b.type)) + "." + std::to_string(b.hex) + "." +
                 std::to_string(int(b.unit_type)) + "." + std::to_string(int(b.upgrade));
        }
    }
    if (a.send_back) {
        s += ",sb" + std::to_string(a.send_back->size());
        for (int i = 0; i < a.send_back->size(); ++i) {
            const oo::SendBackEntry& e = (*a.send_back)[i];
            s += "/" + std::to_string(e.origin_hex) + "." + std::to_string(e.units[0]) + "." +
                 std::to_string(e.units[1]) + "." + std::to_string(e.units[2]);
        }
    }
    return s;
}

// Owns the buffers an answer points into, so a set of answers can outlive the
// call that produced them.
struct AnswerSlot {
    oo::ChosenBuyActions buy;
    oo::SendBack send_back;
    oo::DecisionAnswer answer;
};

}  // namespace

int main(int argc, char** argv) {
    if (!oo::neural_available()) {
        std::printf("test_neural: no LibTorch in this build, skipping\n");
        return 0;
    }

    // A tiny checked-in TorchScript module (d=32, 1 block, r5f4 - 250 KB). It is
    // self-contained, so this covers the ENGINE side of the path - loading,
    // batching, masking, decoding - independently of what train/model.py
    // currently looks like. Regenerate with:
    //   python train/export.py --init --out engine/tests/data/tiny_policy.ts \
    //       --radius 5 --factions 4 --d-model 32 --blocks 1 --heads 2
    const std::string weights = argc > 1 ? argv[1] : "tests/data/tiny_policy.ts";
    const int radius = 5, factions = 4;

    oo::NeuralConfig ncfg;
    ncfg.weights = weights;
    ncfg.radius = radius;
    ncfg.num_factions = factions;
    ncfg.cuda = false;
    ncfg.fp16 = false;

    std::string err;
    auto policy = oo::make_neural_policy(ncfg, err);
    check(policy != nullptr, "the policy builds: " + err);
    if (!policy) return 1;
    std::printf("  %s\n", oo::neural_describe(*policy).c_str());

    // --- a real mid-game board, plus an injected battle ----------------------
    auto state = std::make_unique<oo::GameState>();
    oo::create_initial_state(*state, radius, factions, 5150);
    {
        oo::AgentSet agents;
        oo::build_agents(agents, oo::AgentKind::kGreedy, factions, 5150);
        oo::Rng rng(5150);
        oo::SetupDecisions sd = oo::make_setup_decisions(agents);
        oo::run_city_setup(*state, sd, rng);
        oo::TurnDecisions td = oo::make_turn_decisions(agents);
        for (int t = 0; t < 6; ++t) oo::run_turn(*state, td, rng);
    }
    // Battles are rare mid-turn, so target and rectification would otherwise go
    // untested - and those are exactly the two heads that read a single hex.
    int battle_hex = -1;
    for (int h = 0; h < state->num_hexes; ++h) {
        if (state->army_faction[h] == oo::NO_FACTION && state->passable(h)) {
            battle_hex = h;
            break;
        }
    }
    check(battle_hex >= 0, "found a hex to stage a battle on");
    if (battle_hex >= 0) {
        oo::Battle& b = state->new_battle(battle_hex);
        b.round = 1;
        b.nslots = 2;
        b.slots[0] = oo::BattleSlot{static_cast<int32_t>(battle_hex), {3, 1, 0}, 0, true};
        b.slots[1] = oo::BattleSlot{static_cast<int32_t>(battle_hex), {2, 0, 1}, 1, false};
    }

    // --- one request of every kind, across seats -----------------------------
    std::vector<oo::DecisionRequest> reqs;
    std::vector<oo::LegalMask> masks(static_cast<size_t>(factions) * 2);
    std::vector<oo::LegalBuyActions> buys(static_cast<size_t>(factions));
    std::vector<std::array<bool, oo::MAX_HEXES>> place(1);
    std::vector<int16_t> pool;

    for (int f = 0; f < factions; ++f) {
        oo::legal_movement_mask(*state, f, masks[static_cast<size_t>(f)]);
        oo::legal_cavalry_mask(*state, f, masks[static_cast<size_t>(factions + f)]);
        oo::get_legal_buy_actions(*state, f, buys[static_cast<size_t>(f)]);

        oo::DecisionRequest m;
        m.state = state.get();
        m.faction = f;

        m.kind = oo::DecisionKind::kMovement;
        m.step = f % 3;
        m.legal_mask = &masks[static_cast<size_t>(f)];
        reqs.push_back(m);

        m.kind = oo::DecisionKind::kCavalry;
        m.legal_mask = &masks[static_cast<size_t>(factions + f)];
        reqs.push_back(m);

        m.kind = oo::DecisionKind::kBuy;
        m.legal_mask = nullptr;
        m.legal_buy = &buys[static_cast<size_t>(f)];
        reqs.push_back(m);
        m.legal_buy = nullptr;

        // A different query hex per seat: if a gather ever reads a fixed row or
        // the wrong batch entry, differing hexes are what exposes it.
        m.kind = oo::DecisionKind::kResource;
        m.hex = (f * 7 + 3) % state->num_hexes;
        reqs.push_back(m);

        if (battle_hex >= 0 && f < 2) {
            m.kind = oo::DecisionKind::kTarget;
            m.hex = battle_hex;
            reqs.push_back(m);

            m.kind = oo::DecisionKind::kRectification;
            m.winner = f;
            m.cap = oo::MAX_STACK_SIZE;
            reqs.push_back(m);
            m.winner = -1;
        }
        m.hex = -1;

        m.kind = oo::DecisionKind::kSwap;
        m.leftover_hex = (f * 5) % state->num_hexes;
        m.placer_hex = (f * 11 + 2) % state->num_hexes;
        reqs.push_back(m);
        m.leftover_hex = m.placer_hex = -1;
    }
    {
        oo::legal_placement_mask(*state, place[0].data());
        oo::DecisionRequest m;
        m.state = state.get();
        m.faction = 0;
        m.kind = oo::DecisionKind::kPlacement;
        m.legal_hexes = place[0].data();
        reqs.push_back(m);

        for (int h = 0; h < state->num_hexes && pool.size() < 5; ++h) {
            if (state->city_owner[h] != oo::NO_FACTION) pool.push_back(static_cast<int16_t>(h));
        }
        m.kind = oo::DecisionKind::kDraft;
        m.legal_hexes = nullptr;
        m.pool = pool.data();
        m.pool_size = static_cast<int>(pool.size());
        reqs.push_back(m);
    }
    const int n = static_cast<int>(reqs.size());
    check(n >= 25, "the request set covers every kind across several seats");
    check(pool.size() >= 2, "the draft pool is non-trivial");

    auto run = [&](const std::vector<int>& order, int chunk) {
        std::vector<AnswerSlot> slots(static_cast<size_t>(n));
        for (size_t i = 0; i < slots.size(); ++i) {
            slots[i].buy.clear();
            slots[i].send_back.clear();
            slots[i].answer.buy = &slots[i].buy;
            slots[i].answer.send_back = &slots[i].send_back;
        }
        for (int start = 0; start < n; start += chunk) {
            const int m = std::min(chunk, n - start);
            std::vector<oo::DecisionRequest> sub(static_cast<size_t>(m));
            std::vector<oo::DecisionAnswer> out(static_cast<size_t>(m));
            for (int i = 0; i < m; ++i) {
                sub[static_cast<size_t>(i)] = reqs[static_cast<size_t>(order[start + i])];
                out[static_cast<size_t>(i)] = slots[static_cast<size_t>(order[start + i])].answer;
            }
            policy->evaluate(sub.data(), out.data(), m);
            for (int i = 0; i < m; ++i) {
                slots[static_cast<size_t>(order[start + i])].answer = out[static_cast<size_t>(i)];
            }
        }
        std::vector<std::string> fps(static_cast<size_t>(n));
        for (int i = 0; i < n; ++i) {
            fps[static_cast<size_t>(i)] =
                fingerprint(slots[static_cast<size_t>(i)].answer, reqs[static_cast<size_t>(i)].kind);
        }
        return fps;
    };

    std::vector<int> forward(static_cast<size_t>(n));
    for (int i = 0; i < n; ++i) forward[static_cast<size_t>(i)] = i;
    std::vector<int> reverse(forward.rbegin(), forward.rend());

    const auto base = run(forward, n);          // one batch of everything
    const auto singles = run(forward, 1);       // one position at a time
    const auto reversed = run(reverse, n);      // same batch, opposite order
    const auto chunked = run(forward, 3);       // ragged batches, last one short

    int d_single = 0, d_rev = 0, d_chunk = 0;
    for (int i = 0; i < n; ++i) {
        const size_t k = static_cast<size_t>(i);
        if (base[k] != singles[k]) ++d_single;
        if (base[k] != reversed[k]) ++d_rev;
        if (base[k] != chunked[k]) ++d_chunk;
    }
    check(d_single == 0, "an answer is the same alone as in a full batch (" +
                             std::to_string(d_single) + " differ)");
    check(d_rev == 0, "an answer does not depend on its position in the batch (" +
                          std::to_string(d_rev) + " differ)");
    check(d_chunk == 0, "an answer does not depend on the batch size (" +
                            std::to_string(d_chunk) + " differ)");

    // The fingerprints must not be all-identical, or the comparisons above would
    // pass for a policy that answered the same thing to everything.
    {
        std::vector<std::string> uniq(base);
        std::sort(uniq.begin(), uniq.end());
        uniq.erase(std::unique(uniq.begin(), uniq.end()), uniq.end());
        check(uniq.size() >= 8, "answers actually vary across requests (" +
                                    std::to_string(uniq.size()) + " distinct)");
    }

    // --- a mismatched checkpoint must be REFUSED -------------------------------
    // Not played. A silent mismatch reshapes every hex token and the net plays
    // like noise while nothing reports an error, which is exactly the failure
    // the exporter's stamped attributes exist to prevent (§12.4).
    {
        oo::NeuralConfig bad = ncfg;
        bad.radius = 7;
        std::string e;
        auto p = oo::make_neural_policy(bad, e);
        check(p == nullptr && e.find("radius") != std::string::npos,
              "a radius mismatch is refused with a clear message, got: " + e);
    }
    {
        oo::NeuralConfig bad = ncfg;
        bad.num_factions = 8;
        std::string e;
        auto p = oo::make_neural_policy(bad, e);
        check(p == nullptr, "a faction-count mismatch is refused");
    }
    {
        oo::NeuralConfig bad = ncfg;
        bad.weights.clear();
        std::string e;
        auto p = oo::make_neural_policy(bad, e);
        check(p == nullptr && e.find("export.py") != std::string::npos,
              "missing weights says how to make some, got: " + e);
    }

    // --- and full games still play through ------------------------------------
    {
        oo::RunConfig cfg;
        cfg.radius = radius;
        cfg.num_factions = factions;
        cfg.max_turns = 8;
        const int games = 4;
        std::vector<oo::GameSpec> specs(games);
        for (int g = 0; g < games; ++g) {
            specs[g].seed = 900 + g;
            for (int f = 0; f < factions; ++f) specs[g].seats[f] = oo::AgentKind::kRandom;
        }
        std::vector<oo::GameResult> results(games);
        oo::BatchRunStats stats;
        oo::run_games_batched(cfg, specs.data(), results.data(), games, games, games, *policy,
                              &stats);
        int played = 0;
        for (const oo::GameResult& r : results) {
            if (r.turns > 0) ++played;
            check(r.turns <= cfg.max_turns, "a game respects the turn cap");
        }
        check(played == games, "every game advanced at least one turn");
        std::printf("  %d requests x 4 groupings; %lld evaluations over %d games\n", n,
                    stats.requests, games);
    }

    std::printf("test_neural: %d failures\n", g_failures);
    return g_failures == 0 ? 0 : 1;
}
