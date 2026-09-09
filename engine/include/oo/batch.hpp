// Running many games so their decisions batch together (PLAN.md §10, M9a).
//
// WHY NOT A STATE MACHINE. §10.9 called for making `play_game` resumable - a
// state machine or coroutine over the phase sequence - so N games could sit
// parked at their next decision point. That is a large, invasive rewrite: the
// decision points are not only in `run_turn`'s phase loops but inside
// `resolve_full_battle`'s per-round target queries, so the whole call chain
// would have to become suspendable.
//
// It is also unnecessary, because of a property of the network itself: a
// transformer's batch dimension is fully independent. Attention runs over a
// position's own token sequence, layer norm is per token, and every GEMM
// reduces over features rather than over the batch. So which OTHER positions
// share a batch cannot change a position's result. Batch composition is free to
// vary; only batch SIZE can matter, and only through kernel selection inside the
// GEMM library - which is why inference runs at a fixed padded size (§10.9).
//
// That collapses the requirement. Games do not have to reach decision points in
// a deterministic ORDER, they only have to be evaluated in fixed-size batches.
// So each game runs on its own thread and simply BLOCKS at a decision point;
// threads are the coroutines. Oversubscribing them is fine precisely because
// they are almost always blocked on inference rather than running.
//
// The cost is real and measured rather than assumed: a context switch per
// decision per game thread. See §10.9 for the numbers.

#pragma once

#include "oo/agent.hpp"
#include "oo/policy.hpp"
#include "oo/run.hpp"

#include <memory>

namespace oo {

// Wraps an AgentSet builder as a BatchedPolicy that answers each request
// independently. No batching happens - it exists so the batched path can be
// proven to produce IDENTICAL results to run_games before a network is involved,
// and afterwards as the reference any neural policy is compared against.
//
// One AgentSet is built per game, keyed by the game's seed, exactly as
// run_games does; requests carry no game id, so the mapping is by state pointer.
class ScriptedPolicy : public BatchedPolicy {
public:
    ScriptedPolicy(const RunConfig& cfg, const GameSpec* specs, int num_games);
    ~ScriptedPolicy() override;

    void evaluate(const DecisionRequest* requests, DecisionAnswer* answers, int n) override;

    // Games register the state they are about to play so their agents can be
    // found again from a bare request. Called once per game, before it starts.
    void bind(const GameState* state, int game_index);
    void unbind(const GameState* state);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

struct BatchRunStats {
    long long batches = 0;
    long long requests = 0;
    double mean_batch = 0.0;
    int max_batch = 0;
};

// Plays specs[0..num_games) with `policy` answering every decision, writing
// outcomes into out[0..num_games).
//
// `num_threads` is the number of games IN FLIGHT, not a core count - it is
// deliberately oversubscribed, since a game thread spends most of its life
// blocked waiting for a batch. It is also the ceiling on batch size, because a
// batch can only contain one request per in-flight game.
//
// `batch_size` is the target: the gate flushes when that many requests are
// pending, or earlier when every in-flight game is already waiting (otherwise
// the last few games would deadlock waiting for a batch that can never fill).
void run_games_batched(const RunConfig& cfg, const GameSpec* specs, GameResult* out, int num_games,
                       int num_threads, int batch_size, BatchedPolicy& policy,
                       BatchRunStats* stats = nullptr);

}  // namespace oo
