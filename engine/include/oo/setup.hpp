// Terrain generation and initial state - the port of engine_old/setup.py.
//
// A map is built in rounds. Each round draws a terrain type from a shrinking
// "bag" (weighted by how much of that type is left) and grows a random-sized blob
// of it outward from a hex touching what is already placed, subject to that type's
// shape rules - and never letting a mountain or lake wall part of the board off
// into an unreachable island.
//
// PARITY NOTE - the bag's iteration order is load-bearing. engine_old's BAG_COUNTS
// is a dict in the order plains, lake, mountain, desert, marsh, which is NOT
// terrain-index order, and that order decides which weight pairs with which type
// inside rng.choices. Getting it wrong produces a plausible-looking map that
// diverges from Python's. See kBagOrder.
//
// Capital placement is NOT done here - see placement.hpp.

#pragma once

#include "oo/rng.hpp"
#include "oo/state.hpp"

#include <vector>

namespace oo {

// engine_old/setup.py: BAG_COUNTS, in dict order. See the parity note above.
inline constexpr int kBagSize = 5;
inline constexpr int8_t kBagOrder[kBagSize] = {kPlains, kLake, kMountain, kDesert, kMarsh};
inline constexpr int32_t kBagCounts[kBagSize] = {120, 25, 25, 40, 40};

// engine_old/setup.py: _ROUND_COUNTS, (min, max) hexes per round, indexed by
// terrain type.
inline constexpr int kRoundCountLo[NUM_TERRAIN_TYPES] = {5, 2, 3, 2, 2};
inline constexpr int kRoundCountHi[NUM_TERRAIN_TYPES] = {8, 4, 5, 5, 5};

// One hex placement, in generation order. Feeds terrain_gen_log.json (§1.3), which
// is kept even though its Python viewer is gone - it is the highest-resolution
// record of what generation did, and so the natural place to diff a divergence.
struct TerrainLogEntry {
    int8_t q, r, s;
    int8_t terrain;
    int32_t round;
};

// Fills terrain[0..grid.num_hexes) and, if `log` is non-null, appends every
// individual placement in order.
void generate_terrain(const HexGrid& grid, Rng& rng, int8_t* terrain,
                      std::vector<TerrainLogEntry>* log);

// Generates terrain and seeds starting gold/kill-XP. Leaves city_owner /
// is_capital / city_placer untouched for run_city_setup to fill in.
void create_initial_state(GameState& state, int radius, int num_factions, int64_t seed,
                          std::vector<TerrainLogEntry>* terrain_log = nullptr);

}  // namespace oo
