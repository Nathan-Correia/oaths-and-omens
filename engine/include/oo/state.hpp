// GameState - the port of engine_old/state.py's ArrayState.
//
// A fixed-size, trivially-copyable POD (PLAN.md §4.3). No heap, no vectors, no
// pointers except the non-owning `grid`. That is the whole design: cloning a state
// is a memcpy, which is what tactician_agent's rollouts and any future search need.
//
// BATTLE STORAGE IS SPARSE (M6d). It used to be dense - a full
// [MAX_HEXES][MAX_BATTLE_CONTRIB] set of arrays - because the Python agents read
// `state.battle_faction[hex, k]` straight through the bindings, making the layout
// part of the public contract. That constraint expired once every agent went
// native at M6b, and the dense form was costing 65 207 of the struct's 70 200
// bytes: 93 % of the state, for something that is empty on almost every hex almost
// all the time. Battles are now a small side table indexed by `battle_index`.
//
// `locked` is gone with it: a hex is locked exactly when it has a battle, and
// keeping a separate bool was a second source of truth that could desync.

#pragma once

#include "oo/config.hpp"
#include "oo/grid.hpp"

#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <cstring>

namespace oo {

// One faction's contribution to a battle. Field order is chosen for packing:
// 12 bytes, no padding.
struct BattleSlot {
    int32_t origin;  // hex these units came from - where survivors/overflow retreat to
    int16_t units[NUM_UNIT_TYPES];
    int8_t faction;
    // True iff this slot's units MOVED into the hex to join the fight (attacker,
    // encounter/line-battle participant, or a later reinforcement); false only for
    // a stationary occupant the battle triggered against. Gates the real Archers
    // ability. Deliberately tracked separately from `origin`, which means "where do
    // these units retreat to" and must NOT be used to infer this - a Line Battle's
    // battle_hex tie-break can coincidentally equal one side's own origin hex.
    bool moved;
};

struct Battle {
    BattleSlot slots[MAX_BATTLE_CONTRIB];
    int16_t hex;
    int16_t round;
    uint8_t nslots;  // occupied slots, always contiguous from 0
};

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

    // --- pending battles, sparse ---------------------------------------------
    int16_t battle_index[MAX_HEXES];  // index into `battles`, or -1 for no battle
    // In battle-CREATION order. That order is load-bearing: the per-faction
    // dismount infantry cap tally is shared across every battle resolved in a
    // turn, so which battle is processed first can change outcomes near the cap.
    // engine_old gets this free from dict insertion order.
    Battle battles[MAX_ACTIVE_BATTLES];
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

    bool passable(int hex_index) const { return !kImpassableByTerrain[terrain[hex_index]]; }

    // A hex is locked exactly when a battle is pending on it.
    bool locked(int hex_index) const { return battle_index[hex_index] >= 0; }

    Battle* battle_at(int hex_index) {
        const int i = battle_index[hex_index];
        return i < 0 ? nullptr : &battles[i];
    }
    const Battle* battle_at(int hex_index) const {
        const int i = battle_index[hex_index];
        return i < 0 ? nullptr : &battles[i];
    }

    // Starts a battle on `hex_index` and returns it. Appends to `battles`, which
    // is what makes the array creation-ordered.
    Battle& new_battle(int hex_index) {
        assert(battle_index[hex_index] < 0 && "hex already has a battle");
        // Checked even in release. The bound is provable (config.hpp) and the
        // observed peak is 8, but asserts compile out under NDEBUG and silently
        // running off the end of `battles` would corrupt the whole state. This
        // runs once per battle created, so the check costs nothing measurable.
        if (num_battles >= MAX_ACTIVE_BATTLES) std::abort();
        Battle& b = battles[num_battles];
        b.hex = static_cast<int16_t>(hex_index);
        b.round = 0;
        b.nslots = 0;
        battle_index[hex_index] = num_battles++;
        return b;
    }

    // Removes a battle, preserving the creation order of the rest - Python's
    // list.remove keeps order, and battle order affects outcomes near the shared
    // dismount cap, so this must never be a swap-erase.
    void erase_battle(int hex_index) {
        const int i = battle_index[hex_index];
        if (i < 0) return;
        for (int j = i; j + 1 < num_battles; ++j) {
            battles[j] = battles[j + 1];
            battle_index[battles[j].hex] = static_cast<int16_t>(j);
        }
        --num_battles;
        battle_index[hex_index] = -1;
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
