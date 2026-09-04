// GameState - the port of engine_old/state.py's ArrayState.
//
// A fixed-size, trivially-copyable POD (PLAN.md §4.3). No heap, no vectors, no
// pointers except the non-owning `grid`. That is the whole design: cloning a state
// is a memcpy, which is what tactician_agent's rollouts and any future search need.
//
// Field-for-field with ArrayState, same dtypes, same meanings. Where a Python
// docstring explains a subtlety, it is reproduced here rather than left behind.
//
// BATTLE STORAGE is dense for now - [MAX_HEXES][MAX_BATTLE_CONTRIB] - because the
// Python agents read state.battle_faction[hex], state.battle_units[hex, k] and
// .shape[1] directly through the bindings, making the layout part of the public
// contract until they are native. That constraint expires at M6b; the sparse
// side-table refactor is scheduled as M6d and drops GameState from ~65 KB to
// ~10 KB. `battle_nslots` is the concession in the meantime: it lets internal
// loops iterate used slots instead of scanning all 16, with no layout change.

#pragma once

#include "oo/config.hpp"
#include "oo/grid.hpp"

#include <cstdint>
#include <cstring>

namespace oo {

struct GameState {
    // --- board, per hex ------------------------------------------------------
    int8_t terrain[MAX_HEXES];          // index into TERRAIN_TYPES
    int8_t city_owner[MAX_HEXES];       // NO_FACTION if no city; a capital or an outpost
    bool is_capital[MAX_HEXES];         // only meaningful where city_owner != NO_FACTION
    int8_t outpost_upgrade[MAX_HEXES];  // NO_UPGRADE, or an index into UPGRADE_TYPES; only
                                        // meaningful where city_owner != NO_FACTION and !is_capital
    int8_t city_placer[MAX_HEXES];      // who placed the colourless city here during setup.
                                        // Left populated once setup finishes (harmless) -
                                        // city_owner/is_capital are authoritative from then on.
    int8_t army_faction[MAX_HEXES];     // NO_FACTION if no army
    int16_t army_units[MAX_HEXES][NUM_UNIT_TYPES];
    bool frozen[MAX_HEXES];             // entered a marsh this turn; cannot move again
    bool locked[MAX_HEXES];             // a battle is pending here

    // --- pending battles, per hex x contribution slot ------------------------
    int8_t battle_faction[MAX_HEXES][MAX_BATTLE_CONTRIB];   // NO_FACTION for empty slots
    int32_t battle_origin[MAX_HEXES][MAX_BATTLE_CONTRIB];   // where this slot's units came from
    int16_t battle_units[MAX_HEXES][MAX_BATTLE_CONTRIB][NUM_UNIT_TYPES];
    // True iff this slot's units MOVED into this hex to join the fight (attacker,
    // encounter/line-battle participant, or a later reinforcement); false only for
    // the original stationary occupant the battle triggered against. Gates the real
    // Archers ability to the attacking side. Deliberately tracked separately from
    // battle_origin, which means "where do these units retreat to" and must NOT be
    // used to infer this - a Line Battle's battle_hex tie-break can coincidentally
    // equal one side's own origin hex.
    bool battle_moved[MAX_HEXES][MAX_BATTLE_CONTRIB];
    uint8_t battle_nslots[MAX_HEXES];   // used slots; not in ArrayState, see header note
    int16_t battle_round[MAX_HEXES];

    // Hexes with a pending battle, in battle-CREATION order. Order is load-bearing:
    // the per-faction dismount infantry cap tally is shared across every battle
    // resolved in a turn, so which battle is processed first can change outcomes
    // near the cap. engine_old gets this free from dict insertion order; we track
    // it explicitly. Removal must be an order-preserving compacting erase (Python's
    // list.remove keeps order), never a swap-erase.
    int16_t battle_order[MAX_ACTIVE_BATTLES];
    int16_t num_battles;

    // --- per faction ---------------------------------------------------------
    int32_t capital_settle_order[MAX_FACTIONS];  // order each capital was settled in the
                                                 // draft; -1 before setup. Breaks a tied
                                                 // VP finish (rulebook's Win Condition).
    int32_t gold[MAX_FACTIONS];
    int32_t resources[MAX_FACTIONS][NUM_RESOURCES];
    int32_t kill_xp[MAX_FACTIONS];
    int32_t victory_points[MAX_FACTIONS];
    bool alive[MAX_FACTIONS];  // vestigial: always true, never set false. Kept only for
                               // board_state.json compatibility (PLAN.md §9).

    // --- scalars -------------------------------------------------------------
    int32_t turn_number;
    int32_t num_factions;
    int32_t num_hexes;

    const HexGrid* grid;  // NOT owned; immutable and shared across threads

    // --- helpers -------------------------------------------------------------

    int units_at(int hex_index) const {
        return army_units[hex_index][kInfantry] + army_units[hex_index][kCavalry] +
               army_units[hex_index][kArchers];
    }

    bool passable(int hex_index) const {
        return !kImpassableByTerrain[terrain[hex_index]];
    }

    // Clears a hex's peaceful army entirely. Used where the WHOLE army is absorbed
    // into a battle - not for departures, which only remove what actually moved.
    void clear_army(int hex_index) {
        army_faction[hex_index] = NO_FACTION;
        army_units[hex_index][kInfantry] = 0;
        army_units[hex_index][kCavalry] = 0;
        army_units[hex_index][kArchers] = 0;
        frozen[hex_index] = false;
    }
};

static_assert(std::is_trivially_copyable<GameState>::value,
              "GameState must stay a POD - cloning it is a memcpy, and that is the "
              "entire point of the layout (PLAN.md §4.3)");

// engine_old/state.py: new_empty. Zeroes everything and applies the non-zero
// defaults (NO_FACTION / NO_UPGRADE / -1 settle order / alive).
void new_empty(GameState& state, const HexGrid& grid, int num_factions);

// engine_old/state.py: count_units_in_play - how many of `faction`'s units of one
// type exist, on the board or mid-battle. Prefer a caller-maintained running tally
// over calling this in a loop; this is for one-off checks.
int count_units_in_play(const GameState& state, int faction, int unit_index);

// engine_old/buy.py: count_all_units_in_play - the same, for all three types at
// once, written into out[3].
void count_all_units_in_play(const GameState& state, int faction, int32_t out[NUM_UNIT_TYPES]);

// Number of outposts (non-capital cities) `faction` holds. engine_old/buy.py's
// _outpost_count.
int outpost_count(const GameState& state, int faction);

}  // namespace oo
