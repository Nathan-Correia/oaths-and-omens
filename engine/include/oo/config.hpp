// Compile-time caps and rule constants.
//
// These are constexpr rather than runtime values on purpose (PLAN.md §4.1): it is
// what keeps GameState a fixed-size POD with no heap allocation, which in turn is
// what makes cloning a state a memcpy. tactician_agent's rollouts clone constantly
// today, and any future MCTS will clone far more.
//
// Every value here mirrors one in engine_old/. Where a name differs, the Python
// original is named in a comment.

#pragma once

#include <cstdint>

namespace oo {

// A hex board of radius r holds 3*r*(r+1)+1 hexes: r=7 -> 169, r=8 -> 217,
// r=10 -> 331. Radius 8 is the largest the current terrain generator supports
// (BAG_COUNTS totals 250 hexes, short of a radius-9 board's 271 - see PLAN.md §9);
// the cap is set at 10 so raising that limit later needs no layout change.
inline constexpr int MAX_RADIUS = 10;
inline constexpr int MAX_HEXES = 3 * MAX_RADIUS * (MAX_RADIUS + 1) + 1;  // 331

inline constexpr int MAX_FACTIONS = 10;  // rulebook tops out at 10 players

inline constexpr int NUM_DIRECTIONS = 6;
inline constexpr int NUM_UNIT_TYPES = 3;   // infantry, cavalry, archers
inline constexpr int NUM_RESOURCES = 4;    // wood, iron, clay, fish
inline constexpr int NUM_TERRAIN_TYPES = 5;
inline constexpr int NUM_UPGRADE_TYPES = 3;

// engine_old/state.py: MAX_BATTLE_CONTRIB. A battle's contributions are stored
// padded to this many slots per hex. Battle resolution can extend a battle via
// cavalry dismounts, so this may need revisiting if the assert ever fires.
inline constexpr int MAX_BATTLE_CONTRIB = 16;

// Worst case is one battle per hex. Cheap to size generously (2 bytes each).
inline constexpr int MAX_ACTIVE_BATTLES = MAX_HEXES;

inline constexpr int MAX_STACK_SIZE = 6;  // outside battle this is strict

inline constexpr int8_t NO_FACTION = -1;
inline constexpr int32_t NO_ORIGIN = -1;
inline constexpr int8_t NO_UPGRADE = -1;
inline constexpr int16_t NO_HEX = -1;

// --- indices into the fixed type orders (engine_old/state.py) ---------------

// UNIT_TYPES = ["infantry", "cavalry", "archers"]. This order IS the casualty
// priority used everywhere ("infantry first, then cavalry, then archers").
enum Unit : int { kInfantry = 0, kCavalry = 1, kArchers = 2 };

// TERRAIN_TYPES = ["plains", "mountain", "lake", "desert", "marsh"]
enum Terrain : int8_t { kPlains = 0, kMountain = 1, kLake = 2, kDesert = 3, kMarsh = 4 };

// RESOURCE_TYPES = ["wood", "iron", "clay", "fish"]
enum Resource : int { kWood = 0, kIron = 1, kClay = 2, kFish = 3 };

// UPGRADE_TYPES = ["barracks", "workshop", "temple"]
enum Upgrade : int8_t { kBarracks = 0, kWorkshop = 1, kTemple = 2 };

// engine_old/state.py: IMPASSABLE_BY_TERRAIN, a lookup table rather than a set
// membership test for exactly the reason its docstring gives.
inline constexpr bool kImpassableByTerrain[NUM_TERRAIN_TYPES] = {
    false,  // plains
    true,   // mountain
    true,   // lake
    false,  // desert
    false,  // marsh
};

// engine_old/state.py: SPAWN_CAPS - concurrent, not lifetime.
inline constexpr int32_t kSpawnCaps[NUM_UNIT_TYPES] = {24, 12, 12};

// --- economy (engine_old/collect.py) ----------------------------------------

inline constexpr int32_t kCapitalGold = 3;
inline constexpr int32_t kOutpostGold = 1;
inline constexpr int32_t kOutpostGoldWithBarracks = 2;

inline constexpr int32_t kVpToWin = 50;
inline constexpr int32_t kOutpostVpPerRound = 1;
inline constexpr int32_t kOutpostDestroyVp = 2;
inline constexpr int32_t kTempleVpPerRound = 1;

// --- setup (engine_old/setup.py) --------------------------------------------

inline constexpr int32_t kStartingGold = 50;
inline constexpr int32_t kStartingKillXp = 2;

// --- turn structure (engine_old/turn.py) ------------------------------------

inline constexpr int kMovementSteps = 3;
inline constexpr int kCavalrySteps = 2;

}  // namespace oo
