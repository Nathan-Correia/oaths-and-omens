// Turn orchestration - the port of engine_old/turn.py.
//
// Buy -> Movement (3 steps) -> Cavalry (2 steps) -> Combat -> Terrain -> Collect.
// Collect runs at the END, so a turn's Buy phase always spends what the PREVIOUS
// turn's Collect produced. Turn 1's Buy therefore has only the starting gold,
// matching the rulebook's setup carve-out.
//
// Decisions arrive through a plain struct of callbacks rather than a class
// hierarchy, mirroring engine_old's {faction: callable} dicts. The eventual native
// agents (PLAN.md §6.3) will formalize this into an Agent interface; keeping it a
// POD of function pointers for now avoids committing to a shape that a neural
// policy probably will not want.

#pragma once

#include "oo/actions.hpp"
#include "oo/battle.hpp"
#include "oo/collect.hpp"
#include "oo/rng.hpp"
#include "oo/state.hpp"

namespace oo {

// Each callback receives the faction it is deciding for. Movement callbacks return
// false to move nothing this step (Python's None).
struct TurnDecisions {
    void (*buy)(const GameState&, int faction, const LegalBuyActions& legal, ChosenBuyActions& out,
                void* ctx) = nullptr;
    bool (*movement)(const GameState&, int faction, int step, const LegalMask& legal, Move& out,
                     void* ctx) = nullptr;
    bool (*cavalry)(const GameState&, int faction, int step, const LegalMask& legal, Move& out,
                    void* ctx) = nullptr;
    TargetFn target = nullptr;
    // `cap` is MAX_STACK_SIZE normally, or 0 when the winner has to be evicted
    // from a foreign capital entirely.
    void (*rectification)(const GameState&, int hex_index, int winner, int cap, SendBack& out,
                          void* ctx) = nullptr;
    ResourceChoiceFn resource_choice = nullptr;
    void* ctx = nullptr;
};

// Resolves every pending battle, in battle_order (creation order). That order is
// load-bearing: the dismount infantry-cap tally is shared across every battle in
// the turn, so which battle resolves first can change outcomes near the cap.
//
// When the winner is not the hex's city_owner:
//   - a CAPITAL evicts the winner entirely (cap = 0, ownership unchanged) -
//     you cannot stand units in another player's capital, win or lose;
//   - an OUTPOST is destroyed (owner cleared) and the winner takes
//     kOutpostDestroyVp; they keep standing there as normal.
void run_battle_phase(GameState& state, const TurnDecisions& decisions, Rng& rng);

void run_turn(GameState& state, const TurnDecisions& decisions, Rng& rng);

// None (-1) until some faction has reached kVpToWin. Among those at or above it,
// the strict highest total wins; an exact tie is broken by whoever settled their
// capital LATER during setup (capital_settle_order is a single incrementing
// counter, so no two factions can tie on it).
int get_game_winner(const GameState& state);

// True once the game should stop. The VP condition is a rule and always applies;
// `max_turns` is purely an infra safety net against a runaway game, not part of
// the rules - pass a negative value to disable it.
bool check_game_end(const GameState& state, int max_turns);

}  // namespace oo
