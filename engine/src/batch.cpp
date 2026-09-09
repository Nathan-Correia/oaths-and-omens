#include "oo/batch.hpp"

#include "oo/game.hpp"
#include "oo/grid.hpp"
#include "oo/placement.hpp"
#include "oo/setup.hpp"
#include "oo/turn.hpp"

#include <atomic>
#include <condition_variable>
#include <map>
#include <mutex>
#include <thread>
#include <vector>

namespace oo {

const char* decision_kind_name(DecisionKind k) {
    switch (k) {
        case DecisionKind::kBuy: return "buy";
        case DecisionKind::kMovement: return "movement";
        case DecisionKind::kCavalry: return "cavalry";
        case DecisionKind::kTarget: return "target";
        case DecisionKind::kRectification: return "rectification";
        case DecisionKind::kResource: return "resource";
        case DecisionKind::kPlacement: return "placement";
        case DecisionKind::kDraft: return "draft";
        default: return "swap";
    }
}

// --- ScriptedPolicy ---------------------------------------------------------

struct ScriptedPolicy::Impl {
    RunConfig cfg;
    const GameSpec* specs;
    int num_games;

    std::mutex mu;
    std::map<const GameState*, std::unique_ptr<AgentSet>> by_state;
};

ScriptedPolicy::ScriptedPolicy(const RunConfig& cfg, const GameSpec* specs, int num_games)
    : impl_(std::make_unique<Impl>()) {
    impl_->cfg = cfg;
    impl_->specs = specs;
    impl_->num_games = num_games;
}

ScriptedPolicy::~ScriptedPolicy() = default;

void ScriptedPolicy::bind(const GameState* state, int game_index) {
    auto set = std::make_unique<AgentSet>();
    build_mixed_agents(*set, impl_->specs[game_index].seats, impl_->cfg.num_factions,
                       impl_->specs[game_index].seed);
    std::lock_guard<std::mutex> lock(impl_->mu);
    impl_->by_state[state] = std::move(set);
}

void ScriptedPolicy::unbind(const GameState* state) {
    std::lock_guard<std::mutex> lock(impl_->mu);
    impl_->by_state.erase(state);
}

void ScriptedPolicy::evaluate(const DecisionRequest* requests, DecisionAnswer* answers, int n) {
    for (int i = 0; i < n; ++i) {
        const DecisionRequest& r = requests[i];
        DecisionAnswer& a = answers[i];

        Agent* agent = nullptr;
        {
            // Agents are stateful across a game (legion's claimed set, every RNG
            // stream), so each game must reach its OWN set. The map is read under
            // a lock because games bind and unbind while others are running.
            std::lock_guard<std::mutex> lock(impl_->mu);
            auto it = impl_->by_state.find(r.state);
            if (it == impl_->by_state.end()) continue;
            agent = it->second->get(r.kind == DecisionKind::kRectification ? r.winner : r.faction);
        }

        switch (r.kind) {
            case DecisionKind::kBuy:
                agent->decide_buy(*r.state, r.faction, *r.legal_buy, *a.buy);
                break;
            case DecisionKind::kMovement:
                a.flag = agent->decide_movement(*r.state, r.faction, r.step, *r.legal_mask, a.move);
                break;
            case DecisionKind::kCavalry:
                a.flag = agent->decide_cavalry(*r.state, r.faction, r.step, *r.legal_mask, a.move);
                break;
            case DecisionKind::kTarget:
                a.index = agent->decide_target(*r.state, r.hex, r.faction);
                break;
            case DecisionKind::kRectification:
                agent->decide_rectification(*r.state, r.hex, r.winner, r.cap, *a.send_back);
                break;
            case DecisionKind::kResource:
                a.resource = agent->decide_resource_choice(*r.state, r.faction, r.hex);
                break;
            case DecisionKind::kPlacement:
                a.index = agent->decide_placement(*r.state, r.faction, r.legal_hexes);
                break;
            case DecisionKind::kDraft:
                a.index = agent->decide_draft(*r.state, r.faction, r.pool, r.pool_size);
                break;
            case DecisionKind::kSwap:
                a.flag = agent->decide_swap(*r.state, r.faction, r.leftover_hex, r.placer_faction,
                                            r.placer_hex);
                break;
        }
    }
}

// --- the gate ---------------------------------------------------------------

namespace {

// Where a blocked game thread parks. One per in-flight game.
struct Slot {
    const DecisionRequest* request = nullptr;
    DecisionAnswer* answer = nullptr;
    bool done = false;
};

class BatchGate {
public:
    BatchGate(BatchedPolicy& policy, int batch_size, int in_flight)
        : policy_(policy), batch_size_(batch_size), live_(in_flight) {}

    // Blocks until the policy has answered. Returns false if the run is winding
    // down and this thread should stop.
    void submit_and_wait(const DecisionRequest& req, DecisionAnswer& ans) {
        std::unique_lock<std::mutex> lock(mu_);
        Slot slot;
        slot.request = &req;
        slot.answer = &ans;
        pending_.push_back(&slot);
        ++waiting_;

        // Flush when the batch is full, or when every remaining game is already
        // parked. The second condition is not an optimisation: without it the
        // final few games of a run would wait forever for a batch that can never
        // fill, because there is nobody left to fill it.
        if (static_cast<int>(pending_.size()) >= batch_size_ || waiting_ >= live_) {
            run_batch(lock);
        } else {
            cv_.wait(lock, [&slot] { return slot.done; });
        }
    }

    // A game has finished; it will submit no more requests. Left-behind threads
    // must be re-checked, since `live_` dropping can make them the last ones.
    void retire() {
        std::unique_lock<std::mutex> lock(mu_);
        --live_;
        if (!pending_.empty() && waiting_ >= live_) run_batch(lock);
    }

    BatchRunStats stats() const {
        std::lock_guard<std::mutex> lock(mu_);
        BatchRunStats s = stats_;
        s.mean_batch = s.batches ? double(s.requests) / double(s.batches) : 0.0;
        return s;
    }

private:
    // Called with the lock held. Takes the pending set, evaluates it OUTSIDE the
    // lock so other games can keep arriving, then wakes everyone it answered.
    void run_batch(std::unique_lock<std::mutex>& lock) {
        std::vector<Slot*> batch;
        batch.swap(pending_);
        const int n = static_cast<int>(batch.size());
        waiting_ -= n;
        ++stats_.batches;
        stats_.requests += n;
        if (n > stats_.max_batch) stats_.max_batch = n;

        std::vector<DecisionRequest> reqs(static_cast<size_t>(n));
        std::vector<DecisionAnswer> outs(static_cast<size_t>(n));
        for (int i = 0; i < n; ++i) {
            reqs[static_cast<size_t>(i)] = *batch[static_cast<size_t>(i)]->request;
            outs[static_cast<size_t>(i)] = *batch[static_cast<size_t>(i)]->answer;
        }

        lock.unlock();
        policy_.evaluate(reqs.data(), outs.data(), n);
        lock.lock();

        for (int i = 0; i < n; ++i) {
            *batch[static_cast<size_t>(i)]->answer = outs[static_cast<size_t>(i)];
            batch[static_cast<size_t>(i)]->done = true;
        }
        cv_.notify_all();
    }

    BatchedPolicy& policy_;
    const int batch_size_;
    mutable std::mutex mu_;
    std::condition_variable cv_;
    std::vector<Slot*> pending_;
    int waiting_ = 0;
    int live_ = 0;
    BatchRunStats stats_;
};

// Routes the engine's per-decision callbacks into the gate. One per game.
struct GameCtx {
    BatchGate* gate;
    const GameState* state;
};

DecisionRequest base(const GameCtx& g, DecisionKind kind, int faction) {
    DecisionRequest r;
    r.state = g.state;
    r.kind = kind;
    r.faction = faction;
    return r;
}

void gate_buy(const GameState&, int faction, const LegalBuyActions& legal, ChosenBuyActions& out,
              void* ctx) {
    GameCtx& g = *static_cast<GameCtx*>(ctx);
    DecisionRequest r = base(g, DecisionKind::kBuy, faction);
    r.legal_buy = &legal;
    DecisionAnswer a;
    a.buy = &out;
    g.gate->submit_and_wait(r, a);
}

bool gate_move(DecisionKind kind, const GameCtx& g, int faction, int step, const LegalMask& legal,
               Move& out) {
    DecisionRequest r = base(g, kind, faction);
    r.step = step;
    r.legal_mask = &legal;
    DecisionAnswer a;
    g.gate->submit_and_wait(r, a);
    out = a.move;
    return a.flag;
}

bool gate_movement(const GameState&, int faction, int step, const LegalMask& legal, Move& out,
                   void* ctx) {
    return gate_move(DecisionKind::kMovement, *static_cast<GameCtx*>(ctx), faction, step, legal,
                     out);
}
bool gate_cavalry(const GameState&, int faction, int step, const LegalMask& legal, Move& out,
                  void* ctx) {
    return gate_move(DecisionKind::kCavalry, *static_cast<GameCtx*>(ctx), faction, step, legal, out);
}

int gate_target(const GameState&, int hex_index, int faction, void* ctx) {
    GameCtx& g = *static_cast<GameCtx*>(ctx);
    DecisionRequest r = base(g, DecisionKind::kTarget, faction);
    r.hex = hex_index;
    DecisionAnswer a;
    g.gate->submit_and_wait(r, a);
    return a.index;
}

void gate_rectification(const GameState&, int hex_index, int winner, int cap, SendBack& out,
                        void* ctx) {
    GameCtx& g = *static_cast<GameCtx*>(ctx);
    DecisionRequest r = base(g, DecisionKind::kRectification, winner);
    r.hex = hex_index;
    r.winner = winner;
    r.cap = cap;
    DecisionAnswer a;
    a.send_back = &out;
    g.gate->submit_and_wait(r, a);
}

Resource gate_resource(const GameState&, int faction, int hex_index, void* ctx) {
    GameCtx& g = *static_cast<GameCtx*>(ctx);
    DecisionRequest r = base(g, DecisionKind::kResource, faction);
    r.hex = hex_index;
    DecisionAnswer a;
    g.gate->submit_and_wait(r, a);
    return a.resource;
}

int gate_placement(const GameState&, int faction, const bool* legal, void* ctx) {
    GameCtx& g = *static_cast<GameCtx*>(ctx);
    DecisionRequest r = base(g, DecisionKind::kPlacement, faction);
    r.legal_hexes = legal;
    DecisionAnswer a;
    g.gate->submit_and_wait(r, a);
    return a.index;
}

int gate_draft(const GameState&, int faction, const int16_t* pool, int pool_size, void* ctx) {
    GameCtx& g = *static_cast<GameCtx*>(ctx);
    DecisionRequest r = base(g, DecisionKind::kDraft, faction);
    r.pool = pool;
    r.pool_size = pool_size;
    DecisionAnswer a;
    g.gate->submit_and_wait(r, a);
    return a.index;
}

bool gate_swap(const GameState&, int faction, int leftover, int placer, int placer_hex, void* ctx) {
    GameCtx& g = *static_cast<GameCtx*>(ctx);
    DecisionRequest r = base(g, DecisionKind::kSwap, faction);
    r.leftover_hex = leftover;
    r.placer_faction = placer;
    r.placer_hex = placer_hex;
    DecisionAnswer a;
    g.gate->submit_and_wait(r, a);
    return a.flag;
}

// One game, start to finish, on this thread. Mirrors play_game exactly - the
// only difference is where the decisions come from.
void play_one_batched(const RunConfig& cfg, const GameSpec& spec, GameResult& out, BatchGate& gate,
                      BatchedPolicy& policy, ScriptedPolicy* scripted, int game_index) {
    auto state = std::make_unique<GameState>();
    create_initial_state(*state, cfg.radius, cfg.num_factions, spec.seed);
    if (scripted) scripted->bind(state.get(), game_index);

    GameCtx gctx{&gate, state.get()};

    SetupDecisions sd;
    sd.placement = &gate_placement;
    sd.draft = &gate_draft;
    sd.swap = &gate_swap;
    sd.ctx = &gctx;

    TurnDecisions td;
    td.buy = &gate_buy;
    td.movement = &gate_movement;
    td.cavalry = &gate_cavalry;
    td.target = &gate_target;
    td.rectification = &gate_rectification;
    td.resource_choice = &gate_resource;
    td.ctx = &gctx;

    Rng rng(spec.seed);
    run_city_setup(*state, sd, rng);

    out = GameResult{};
    while (!check_game_end(*state, cfg.max_turns)) {
        run_turn(*state, td, rng);
        ++out.turns;
    }
    out.winner = get_game_winner(*state);
    for (int f = 0; f < cfg.num_factions; ++f) out.victory_points[f] = state->victory_points[f];

    if (scripted) scripted->unbind(state.get());
    (void)policy;
}

}  // namespace

void run_games_batched(const RunConfig& cfg, const GameSpec* specs, GameResult* out, int num_games,
                       int num_threads, int batch_size, BatchedPolicy& policy,
                       BatchRunStats* stats) {
    if (num_games <= 0) return;
    if (num_threads <= 0) num_threads = 1;
    if (num_threads > num_games) num_threads = num_games;
    if (batch_size <= 0) batch_size = num_threads;
    if (batch_size > num_threads) batch_size = num_threads;

    (void)HexGrid::shared(cfg.radius);

    // The gate is told how many games are in flight, not how many remain, so it
    // knows when everyone left is parked.
    BatchGate gate(policy, batch_size, num_threads);
    auto* scripted = dynamic_cast<ScriptedPolicy*>(&policy);

    std::atomic<int> next{0};
    std::vector<std::thread> workers;
    workers.reserve(static_cast<size_t>(num_threads));
    for (int t = 0; t < num_threads; ++t) {
        workers.emplace_back([&] {
            for (;;) {
                const int i = next.fetch_add(1, std::memory_order_relaxed);
                if (i >= num_games) break;
                play_one_batched(cfg, specs[i], out[i], gate, policy, scripted, i);
            }
            // This thread will never submit again. Tell the gate so the games
            // still running are not left waiting on it.
            gate.retire();
        });
    }
    for (std::thread& w : workers) w.join();

    if (stats) *stats = gate.stats();
}

}  // namespace oo
