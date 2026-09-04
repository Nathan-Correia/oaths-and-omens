// Shared agent helpers - the pieces greedy/heuristic/vanguard/marshal reuse.
//
// TIE-BREAKING IS THE WHOLE GAME HERE. Python's `min`/`max` return the FIRST
// extremum and `sorted` is stable, and several of these agents lean on that
// deliberately (vanguard's docstring records that a naive min()-over-direction
// tiebreak was a measured regression). So:
//
//   Python                     C++
//   min(xs, key=f)             first strict improvement only  (`<`, never `<=`)
//   max(xs, key=f)             first strict improvement only  (`>`, never `>=`)
//   sorted(xs, key=f)          std::stable_sort, never std::sort
//
// greedy/heuristic rank mobile armies by size, and the Python originals used
// `np.argsort(-sizes)` - whose order among EQUAL keys is an unreproducible numpy
// implementation artifact. Army sizes are small integers so ties are the common
// case, and it measurably differs from a stable sort in 62% of random 8-element
// arrays and ~100% by 20. Those two call sites now pass kind="stable" (see
// PLAN.md §6.6), making the tie-break plain ascending hex order, which is what
// mobile_hexes_by_size_desc reproduces.

#pragma once

#include "oo/buy.hpp"
#include "oo/grid.hpp"
#include "oo/state.hpp"

namespace oo {

// A list of board coordinates used as movement objectives.
using CoordList = SmallVec<HexCoord, 128>;
using HexList = SmallVec<int16_t, MAX_HEXES>;

// heuristic_agent's _UNIT_POWER: archers and cavalry outvalue plain infantry
// thanks to their battle abilities. Used only for threat/target scoring, never
// for the real battle math.
inline constexpr double kUnitPower[NUM_UNIT_TYPES] = {1.0, 1.2, 1.3};

inline double army_power(const int16_t units[NUM_UNIT_TYPES]) {
    return units[0] * kUnitPower[0] + units[1] * kUnitPower[1] + units[2] * kUnitPower[2];
}

// Hexes where `faction` has an army that may legally move this step, ascending.
void mobile_hexes(const GameState& state, const LegalMask& legal, HexList& out);

// Same, but ranked by descending army size - greedy/heuristic's
// `np.argsort(-sizes)`. See the header note on stability.
void mobile_hexes_by_size_desc(const GameState& state, const LegalMask& legal, HexList& out);

// heuristic_agent's _resource_bonus: +2 for a hex adjacent to both mountain and
// lake, +1 for one of them, -2 for a desert with neither, else 0.
double resource_bonus(const GameState& state, int hex_index);

// Enemy outposts / enemy capitals, as coordinates, ascending by hex index.
void enemy_outpost_coords(const GameState& state, int faction, CoordList& out);
void enemy_capital_coords(const GameState& state, int faction, CoordList& out);

// The faction's own capital, or -1.
int own_capital(const GameState& state, int faction);

// Hexes where this faction could legally found an outpost AND the terrain is
// passable - `eligible_outpost_mask & ~IMPASSABLE`.
void eligible_expansion_hexes(const GameState& state, int faction, HexList& out);

// greedy_agent's _move_toward: walk `ranked_origins` and return the first army
// that can step toward its own nearest target. Returns false if none can.
bool move_toward(const GameState& state, const HexList& ranked_origins, const LegalMask& legal,
                 const CoordList& targets, bool skip_arrived, Move& out);

// vanguard_agent's _direction_tiebreak - a stable pseudo-random ordering so
// equally good directions do not all collapse onto the lowest index.
inline int direction_tiebreak(int origin, int dest, int turn_number) {
    return (origin * 92821 + dest * 68917 + turn_number * 4241) % 1000003;
}

// vanguard_agent's _best_direction: minimise (distance, lands-on-bad-desert,
// lands-on-marsh, tiebreak), then optionally detour around a marsh if a
// non-marsh option costs at most MARSH_DETOUR_TOLERANCE extra hexes.
inline constexpr int kMarshDetourTolerance = 1;
int best_direction(const GameState& state, int origin, const LegalMask& legal,
                   const HexCoord& target, int steps_remaining);

// vanguard_agent's ranked objective pools.
inline constexpr int kExpansionObjectives = 50;
inline constexpr int kAttackObjectives = 25;
inline constexpr double kExpansionResourceWeight = 1.0;

void ranked_expansion_targets(const GameState& state, int faction, CoordList& out);
void ranked_attack_targets(const GameState& state, int faction, CoordList& out);
void all_targets(const GameState& state, int faction, CoordList& out);

// marshal_agent's _greedy_match: repeatedly pair off the closest unmatched
// (origin, target). Sorted by DISTANCE ONLY, so ties break by generation order -
// origin-major, then target - which requires a stable sort.
struct MatchPair {
    int16_t origin;
    HexCoord target;
};
using MatchList = SmallVec<MatchPair, MAX_HEXES>;
void greedy_match(const GameState& state, const HexList& origins, const CoordList& targets,
                  MatchList& out);

}  // namespace oo
