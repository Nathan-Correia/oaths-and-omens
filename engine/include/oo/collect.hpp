// Collect phase - the port of engine_old/collect.py.
//
// The LAST step of each turn (Buy -> Movement -> Combat -> Collect). Because it
// runs at the end, the following turn's Buy phase spends whatever this produced;
// turn 1's Buy has only the starting gold, matching the rulebook's setup carve-out.
//
// Three things happen, in order: gold income, resource income, then the recurring
// per-round victory-point tally. Kept as separate functions so each is
// independently testable, bundled by apply_collect_phase.

#pragma once

#include "oo/state.hpp"

namespace oo {

// Asked only about an outpost adjacent to BOTH a mountain and a lake - every other
// outpost's resource is determined by terrain alone. Mirrors engine_old's
// decide_resource_choice callback.
//
// Python returns the string "iron" or "fish" and treats anything that is not
// exactly "iron" as fish; returning kIron here is the "iron" case, and any other
// value is fish, so the two agree on every input.
using ResourceChoiceFn = Resource (*)(const GameState& state, int faction, int hex_index, void* ctx);

// +3 gold/turn from a faction's capital, +1 per outpost (+2 if that outpost has a
// Barracks). A faction with zero cities gets no income and instead loses one unit -
// unreachable in practice since capitals are permanent, kept for parity.
void apply_gold_income(GameState& state);

// Wood/Iron/Clay/Fish from every outpost; capitals never generate resources.
void apply_resource_income(GameState& state, ResourceChoiceFn choose, void* ctx);

// End-of-round VP: your first outpost earns nothing and each additional one earns
// 1 more per round - max(0, outposts - 1), not a flat per-outpost rate - plus 1 per
// Temple. Capitals don't count. Destroying an enemy outpost is awarded separately
// and immediately in the battle phase (kOutpostDestroyVp).
void apply_victory_points(GameState& state);

void apply_collect_phase(GameState& state, ResourceChoiceFn choose, void* ctx);

}  // namespace oo
