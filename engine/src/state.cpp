#include "oo/state.hpp"

#include <cassert>
#include <cstring>

namespace oo {

void new_empty(GameState& state, const HexGrid& grid, int num_factions) {
    assert(num_factions > 0 && num_factions <= MAX_FACTIONS);
    std::memset(&state, 0, sizeof(GameState));

    // Everything else is zero-initialised above; only the non-zero defaults from
    // engine_old/state.py's new_empty need writing. They are set across the FULL
    // fixed arrays, not just [0, num_hexes), so that two states with the same
    // logical contents always compare equal byte-for-byte - which is what lets
    // parity checks and future transposition hashing memcmp the struct directly.
    std::memset(state.city_owner, NO_FACTION, sizeof(state.city_owner));
    std::memset(state.city_placer, NO_FACTION, sizeof(state.city_placer));
    std::memset(state.army_faction, NO_FACTION, sizeof(state.army_faction));
    std::memset(state.outpost_upgrade, NO_UPGRADE, sizeof(state.outpost_upgrade));

    for (int i = 0; i < MAX_HEXES; ++i) state.battle_index[i] = -1;
    // The battle table is left zeroed rather than filled with NO_FACTION: only
    // the first `num_battles` entries are ever read, and each is initialised by
    // new_battle(). Zeroing the whole array keeps memcmp-equality meaningful.

    for (int f = 0; f < MAX_FACTIONS; ++f) {
        state.capital_settle_order[f] = -1;
        state.alive[f] = true;
    }

    state.num_factions = num_factions;
    state.num_hexes = grid.num_hexes();
    state.grid = &grid;
    state.turn_number = 0;
    state.num_battles = 0;
}

int count_units_in_play(const GameState& state, int faction, int unit_index) {
    int total = 0;
    for (int i = 0; i < state.num_hexes; ++i) {
        if (state.army_faction[i] == faction) total += state.army_units[i][unit_index];
    }
    for (int b = 0; b < state.num_battles; ++b) {
        const Battle& battle = state.battles[b];
        for (int k = 0; k < battle.nslots; ++k) {
            if (battle.slots[k].faction == faction) total += battle.slots[k].units[unit_index];
        }
    }
    return total;
}

void count_all_units_in_play(const GameState& state, int faction, int32_t out[NUM_UNIT_TYPES]) {
    out[kInfantry] = out[kCavalry] = out[kArchers] = 0;
    for (int i = 0; i < state.num_hexes; ++i) {
        if (state.army_faction[i] == faction) {
            out[kInfantry] += state.army_units[i][kInfantry];
            out[kCavalry] += state.army_units[i][kCavalry];
            out[kArchers] += state.army_units[i][kArchers];
        }
    }
    for (int b = 0; b < state.num_battles; ++b) {
        const Battle& battle = state.battles[b];
        for (int k = 0; k < battle.nslots; ++k) {
            if (battle.slots[k].faction != faction) continue;
            out[kInfantry] += battle.slots[k].units[kInfantry];
            out[kCavalry] += battle.slots[k].units[kCavalry];
            out[kArchers] += battle.slots[k].units[kArchers];
        }
    }
}

int outpost_count(const GameState& state, int faction) {
    int n = 0;
    for (int i = 0; i < state.num_hexes; ++i) {
        if (state.city_owner[i] == faction && !state.is_capital[i]) ++n;
    }
    return n;
}

}  // namespace oo
