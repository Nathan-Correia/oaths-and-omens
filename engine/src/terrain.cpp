#include "oo/terrain.hpp"

namespace oo {

void apply_terrain_effects(GameState& state) {
    for (int h = 0; h < state.num_hexes; ++h) {
        const bool needs_desert_loss = state.army_faction[h] != NO_FACTION &&
                                       state.terrain[h] == kDesert &&
                                       state.city_owner[h] == NO_FACTION;
        if (!needs_desert_loss) continue;

        for (int ut = 0; ut < NUM_UNIT_TYPES; ++ut) {  // infantry -> cavalry -> archers
            if (state.army_units[h][ut] > 0) {
                state.army_units[h][ut] -= 1;
                break;
            }
        }
        if (state.units_at(h) == 0) {
            state.army_faction[h] = NO_FACTION;
            state.frozen[h] = false;
        }
    }

    // Unfreeze whatever is left standing. Testing army_faction again here (rather
    // than reusing the value from before the desert pass) matters: a hex the desert
    // loss just emptied must NOT be touched, mirroring engine_old/terrain.py's
    // `continue` that skips the unfreeze check for a hex wiped out in the same pass.
    for (int h = 0; h < state.num_hexes; ++h) {
        if (state.army_faction[h] != NO_FACTION && state.frozen[h]) {
            state.frozen[h] = false;
        }
    }
}

}  // namespace oo
