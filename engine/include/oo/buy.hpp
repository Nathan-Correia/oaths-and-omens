// Buy phase - the port of engine_old/buy.py.
//
// Four kinds of purchase, each atomic and worth one unit/building:
//   buy_infantry       2 gold at an owned city (capital or outpost)
//   convert_to_special 1 kill-XP + 1 gold turns one of your infantry into
//                      cavalry or archers - the only way to get either
//   build_outpost      3 gold + consumes 1 unit standing on the target hex
//   upgrade_outpost    that upgrade's resource cost; converting from one upgrade
//                      to another costs the new one's FULL price, no credit
//
// Two limits are per-turn-BATCH properties rather than a single action's own
// legality, so they live in apply_buy_phase and NOT in get_legal_buy_actions,
// matching engine_old:
//   - at most 1 infantry recruited per outpost per turn, unless it has a Barracks
//     (capitals have no such cap either way)
//   - at most 1 outpost action (build or upgrade, combined) per faction per turn
//
// RULE CHANGE carried over from engine_old: the no-adjacent-enemy requirement
// applies only at an outpost. A capital can always recruit regardless of what is
// standing next to it.

#pragma once

#include "oo/actions.hpp"
#include "oo/state.hpp"

namespace oo {

inline constexpr int32_t kInfantryCost = 2;
inline constexpr int32_t kOutpostCost = 3;
inline constexpr int kOutpostCap = 6;

// "not within 2 tiles of your own capital" etc. - expressed as minimum distances.
inline constexpr int kOutpostMinDistOwnCapital = 3;
inline constexpr int kOutpostMinDistEnemyCapital = 2;
inline constexpr int kOutpostMinDistOtherOutpost = 2;

// Resource cost per upgrade, indexed [upgrade][resource]. engine_old's
// UPGRADE_COSTS. Barracks: 2 fish 4 wood. Workshop: 2 iron 2 clay 4 wood.
// Temple: 2 fish 2 iron 2 clay 4 wood.
inline constexpr int32_t kUpgradeCosts[NUM_UPGRADE_TYPES][NUM_RESOURCES] = {
    /* barracks */ {4, 0, 0, 2},
    /* workshop */ {4, 2, 2, 0},
    /* temple   */ {4, 2, 2, 2},
};

// Where `faction` could legally found a new outpost, for every hex at once.
// engine_old vectorized this because the per-hex version was 88% of a whole
// game's runtime; here it is three passes over the precomputed distance table.
void eligible_outpost_mask(const GameState& state, int faction, bool out[MAX_HEXES]);

bool can_build_outpost(const GameState& state, int hex_index, int faction);

// Order matters and is part of the contract - agents index into this list, and
// the parity tests compare it element by element.
void get_legal_buy_actions(const GameState& state, int faction, LegalBuyActions& out);

// Applies each faction's chosen actions in order, enforcing the two per-turn
// batch caps described above. `chosen[f]` is faction f's list.
void apply_buy_phase(GameState& state, const ChosenBuyActions chosen[MAX_FACTIONS]);

}  // namespace oo
