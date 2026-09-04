// Movement - the port of engine_old/movement.py.
//
// One simultaneous step: every faction may move at most one army one hex. The
// regular phase moves the WHOLE army; the cavalry phase moves exactly the cavalry
// count, leaving all infantry and archers behind. Never an arbitrary split.
//
// Three ways a battle starts here (all resolved later, in the battle phase):
//   Attack/Defense  moving into a tile held by another faction
//   Encounter       two or more armies moving into the same empty tile
//   Line Battle     two adjacent armies moving into each other (a swap)
// Plus: arriving at a hex owned by a foreign faction ALWAYS forces a battle, even
// with no defending army - capitals and outposts are no longer capturable by
// walking in. What happens to the winner afterwards is turn.cpp's job.
//
// PORTED QUIRKS - deliberately preserved, see revert_departure in the .cpp:
// reverting an overstacked move can start a battle on the mover's own origin, or
// recreate a peaceful army on a hex an unrelated battle locked this same step.
// These are engine_old's behaviour and parity is worth more than tidiness; PLAN.md
// §9 tracks revisiting them once the port is green.

#pragma once

#include "oo/actions.hpp"
#include "oo/rng.hpp"
#include "oo/state.hpp"

namespace oo {

// legal_mask[h][d]: faction's whole army at h may move to h's neighbour in
// direction d this step.
void legal_movement_mask(const GameState& state, int faction, LegalMask& out);

// Same, but only for hexes with cavalry present.
void legal_cavalry_mask(const GameState& state, int faction, LegalMask& out);

// Applies one simultaneous step in place. `cavalry_only` must match the mask the
// actions were built from. `rng` is consumed ONLY for a Line Battle's exact-tie
// coin flip - but that draw's position in the sequence is load-bearing, so pass
// the same generator used for the rest of the turn.
void apply_movement_step(GameState& state, const MoveActions& actions, Rng& rng, bool cavalry_only);

}  // namespace oo
