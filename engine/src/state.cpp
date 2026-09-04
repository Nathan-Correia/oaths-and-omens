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
    std::memset(state.battle_faction, NO_FACTION, sizeof(state.battle_faction));

    for (int i = 0; i < MAX_HEXES; ++i) {
        for (int k = 0; k < MAX_BATTLE_CONTRIB; ++k) {
            state.battle_origin[i][k] = NO_ORIGIN;
        }
    }
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
    // Battle contributions: only hexes with a live battle can hold any, so walk
    // battle_order rather than every hex.
    for (int b = 0; b < state.num_battles; ++b) {
        const int h = state.battle_order[b];
        for (int k = 0; k < state.battle_nslots[h]; ++k) {
            if (state.battle_faction[h][k] == faction) total += state.battle_units[h][k][unit_index];
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
        const int h = state.battle_order[b];
        for (int k = 0; k < state.battle_nslots[h]; ++k) {
            if (state.battle_faction[h][k] == faction) {
                out[kInfantry] += state.battle_units[h][k][kInfantry];
                out[kCavalry] += state.battle_units[h][k][kCavalry];
                out[kArchers] += state.battle_units[h][k][kArchers];
            }
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
