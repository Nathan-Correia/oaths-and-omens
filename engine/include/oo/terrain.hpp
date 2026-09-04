// Terrain effects - the port of engine_old/terrain.py.
//
// Applied once at the end of each full turn, after movement, cavalry movement and
// battles have all resolved:
//   - Desert: an army ending the turn on a desert hex with no city on it loses 1
//     unit (infantry -> cavalry -> archers priority).
//   - Marsh:  armies frozen this turn (by entering a marsh hex) unfreeze, ready to
//     move again next turn.

#pragma once

#include "oo/state.hpp"

namespace oo {

void apply_terrain_effects(GameState& state);

}  // namespace oo
