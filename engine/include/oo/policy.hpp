// Batched decision interface (PLAN.md §10, M9a).
//
// A neural policy cannot answer one decision at a time: a forward pass costs the
// same for 1 position as for 256, so evaluating singly wastes the GPU by two
// orders of magnitude. Something has to gather decisions from many games in
// flight and answer them together. This is the interface that gathering targets.
//
// A request is deliberately a flat struct covering all nine decision kinds rather
// than nine separate queues. The engine asks for whatever it needs next, and a
// policy that batches does not want to care which kind it is until it sorts the
// batch; a policy that does not batch (the scripted adapter) just switches on the
// kind. Fields not relevant to a kind are unset.
//
// Pointers in a request are BORROWED and valid only for the duration of the
// evaluate() call: they point into the calling game thread's own stack, which is
// blocked waiting. Nothing here is copied, which is the point - a
// ChosenBuyActions is 4 KB and a batch of 1024 would otherwise be 4 MB of
// pointless memcpy per decision point.

#pragma once

#include "oo/actions.hpp"
#include "oo/collect.hpp"
#include "oo/state.hpp"

#include <cstdint>

namespace oo {

enum class DecisionKind : uint8_t {
    kBuy,
    kMovement,
    kCavalry,
    kTarget,
    kRectification,
    kResource,
    kPlacement,
    kDraft,
    kSwap,
};

inline constexpr int kNumDecisionKinds = 9;
const char* decision_kind_name(DecisionKind k);

struct DecisionRequest {
    const GameState* state = nullptr;
    DecisionKind kind = DecisionKind::kMovement;
    int faction = -1;

    int step = 0;  // movement / cavalry sub-step

    // The hex the question is about: the battle hex for target and
    // rectification, the outpost for a resource choice.
    int hex = -1;

    int winner = -1;  // rectification
    int cap = 0;      // rectification

    int leftover_hex = -1;    // swap
    int placer_faction = -1;  // swap
    int placer_hex = -1;      // swap

    // Legality, exactly as the engine already computes it. Whichever one applies
    // to `kind` is set; the rest stay null.
    const LegalBuyActions* legal_buy = nullptr;
    const LegalMask* legal_mask = nullptr;
    const bool* legal_hexes = nullptr;  // [MAX_HEXES], placement
    const int16_t* pool = nullptr;      // draft
    int pool_size = 0;
};

struct DecisionAnswer {
    // movement / cavalry: whether a move was made at all. swap: accept.
    bool flag = false;
    Move move{};

    // target / placement / draft: the chosen index.
    int index = -1;

    Resource resource = kFish;

    // Written in place by the policy; these live on the game thread's stack.
    ChosenBuyActions* buy = nullptr;
    SendBack* send_back = nullptr;
};

// Answers a whole batch at once. Implementations must fill out[i] for every i.
//
// CALLED CONCURRENTLY. The gate deliberately evaluates outside its lock so
// games can keep arriving while a batch runs, so several threads may be inside
// evaluate() at once, each with requests from different games. An
// implementation must therefore be reentrant: no shared mutable scratch, no
// per-game state of its own. A shared staging buffer here is a data race that
// surfaces as a stack-buffer-overrun crash, not as a wrong answer.
class BatchedPolicy {
public:
    virtual ~BatchedPolicy() = default;
    virtual void evaluate(const DecisionRequest* requests, DecisionAnswer* answers, int n) = 0;

    // Called once per game, on that game's own thread, before and after it
    // plays. A stateless policy (a network) ignores both; a policy holding
    // per-game state (scripted agents, whose RNGs and memories persist across a
    // whole game) uses them to build and drop that state.
    //
    // These are virtuals rather than a dynamic_cast in the runner because a
    // COMPOSITE policy - part scripted, part neural, which is what head-to-head
    // evaluation needs - has to forward them to whichever halves care.
    virtual void on_game_start(const GameState*, int /*game_index*/) {}
    virtual void on_game_end(const GameState*) {}
};

}  // namespace oo
