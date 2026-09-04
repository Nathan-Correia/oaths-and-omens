// Native agent interface (PLAN.md §6.3).
//
// One instance per faction per game, each owning its own Rng - mirroring
// engine_old's `rngs = {f: random.Random(seed * 1_000_003 + f) ...}` in every
// make_X_agents. Agents are STATEFUL across turns (legion_agent closes over a
// `claimed` set that persists all game, and every RNG stream persists), so they
// are per-game objects, not shared singletons, and must be cheap to construct -
// §7's thread pool wants one full set per thread.

#pragma once

#include "oo/actions.hpp"
#include "oo/collect.hpp"
#include "oo/placement.hpp"
#include "oo/rng.hpp"
#include "oo/state.hpp"
#include "oo/turn.hpp"

#include <memory>

namespace oo {

class Agent {
public:
    virtual ~Agent() = default;

    virtual void decide_buy(const GameState& state, int faction, const LegalBuyActions& legal,
                            ChosenBuyActions& out) = 0;
    virtual bool decide_movement(const GameState& state, int faction, int step,
                                 const LegalMask& legal, Move& out) = 0;
    virtual bool decide_cavalry(const GameState& state, int faction, int step,
                                const LegalMask& legal, Move& out) = 0;
    virtual int decide_target(const GameState& state, int hex_index, int faction) = 0;
    virtual void decide_rectification(const GameState& state, int hex_index, int winner, int cap,
                                      SendBack& out) = 0;
    virtual Resource decide_resource_choice(const GameState& state, int faction, int hex_index) = 0;
    virtual int decide_placement(const GameState& state, int faction,
                                 const bool legal[MAX_HEXES]) = 0;
    virtual int decide_draft(const GameState& state, int faction, const int16_t* pool,
                             int pool_size) = 0;
    virtual bool decide_swap(const GameState& state, int faction, int leftover_hex,
                             int placer_faction, int placer_hex) = 0;

    // Reserved for action cards, which are in the rulebook but absent from the
    // engine (PLAN.md §9). A no-op default costs nothing now and means adding
    // them later does not have to touch every agent - the expensive part of
    // retrofitting a tenth decision point.
    virtual void decide_play_cards(const GameState&, int /*faction*/) {}

    Rng rng;
};

// The per-faction agents for one game.
struct AgentSet {
    std::unique_ptr<Agent> agents[MAX_FACTIONS];
    int num_factions = 0;

    Agent* get(int faction) const { return agents[faction].get(); }
};

// Adapters turning an AgentSet into the plain callback structs the engine takes.
// The engine deliberately does not know about Agent (PLAN.md §6.3 keeps the
// callback path alive so a future NN policy can plug in the same way).
TurnDecisions make_turn_decisions(const AgentSet& agents);
SetupDecisions make_setup_decisions(const AgentSet& agents);

// Agent kinds implemented natively so far.
enum class AgentKind {
    kRandom, kGreedy, kHeuristic, kVanguard, kMarshal,       // M6a
    kTurtle, kDenier, kWarlord, kLegion, kHussar, kSentinel,  // M6b leaves
    kTactician,                                              // M6b, the search agent
};

// `seed` is the per-GAME seed; each faction's generator is seeded
// seed * 1_000_003 + faction, exactly as engine_old does.
void build_agents(AgentSet& out, AgentKind kind, int num_factions, int64_t seed);

// Returns false for an unrecognised name.
bool agent_kind_from_name(const char* name, AgentKind& out);

}  // namespace oo
