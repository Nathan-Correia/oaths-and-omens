"""
GreedyAgent for engine: a single-minded outpost rush.

Buy phase prioritizes its one outpost action for the turn (build_outpost or
upgrade_outpost, combined 1/turn - see engine/buy.py): upgrading an outpost
it already holds comes before building a new one, since it's pure value-add
on ground already secured (see greedy_buy/_UPGRADE_PRIORITY). Whatever's
left then gets spent aggressively - infantry with leftover gold, all banked
kill-XP converted toward whichever of cavalry/archers it currently has
fewer of.

Movement/cavalry phases have two priorities, strictly ordered: while this
faction still has room for more outposts (under OUTPOST_CAP) and at
least one hex anywhere on the board is currently legal to found one on,
every decision funnels armies toward the single such hex closest to this
faction's own capital - once one army settles there the buy phase builds
on it (see greedy_buy), and the next call just finds the new nearest
spot, so expansion keeps going one outpost at a time rather than
scattering everywhere at once. Only once that's exhausted (cap reached,
or literally nowhere legal left) do armies switch to attacking: straight
at the nearest enemy outpost to destroy it (2 VP for the kill, plus it
denies that opponent their per-round outpost VP going forward), or, if
none exist yet, the nearest enemy capital instead, just to keep units
advancing rather than sitting idle.

No retreat logic, no threat-aware pathing around stronger armies en
route - always takes whichever legal step most closes distance to the
current target, for better or worse. That single-mindedness is what
"greedy" means here, not tactical caution.

decide_target/decide_rectification are reused as-is from random_agent.py
- neither needs to be smart for this agent's strategy to work.
decide_resource_choice gets a one-line heuristic (greedy_resource_choice:
prefer whichever of iron/fish is scarcer) rather than being reused as-is,
since a uniform coin flip there would just be random_agent's policy under
another name. Setup-phase decisions (placement/draft/swap) get their own
greedy policy below: farthest-point placement/drafting, since a uniform
coin flip would be an odd fit for an agent named "greedy".
"""

import random

import numpy as np

from .random_agent import random_rectification, random_target
from engine.buy import INFANTRY_COST, OUTPOST_CAP, eligible_outpost_mask, _outpost_count
from engine.geometry import hex_distance
from engine.state import IMPASSABLE_BY_TERRAIN, NO_FACTION, NO_UPGRADE, RESOURCE_TO_INDEX, count_units_in_play

# Which unit type to sacrifice first when a hex offers a choice of more
# than one for building an outpost there.
_OUTPOST_UNIT_PRIORITY = ("infantry", "cavalry", "archers")

# Which upgrade to grab first when more than one is affordable: Temple
# first (direct VP - the actual win condition), then Barracks (compounds
# economy), then Workshop (compounds resources) - a "greedy" ranking by
# how directly each pays off.
_UPGRADE_PRIORITY = ("temple", "barracks", "workshop")


def greedy_buy(state, faction, legal, rng):
    outpost_actions = [a for a in legal if a["type"] == "build_outpost"]
    # Only ever gives a bare outpost its FIRST upgrade - get_legal_buy_actions
    # also offers converting an already-upgraded outpost to a different
    # upgrade, but re-paying full price to swap isn't worth it for a
    # strategy this simple, so those are filtered out here.
    upgrade_actions = [
        a for a in legal
        if a["type"] == "upgrade_outpost" and state.outpost_upgrade[a["hex"]] == NO_UPGRADE
    ]
    infantry_actions = [a for a in legal if a["type"] == "buy_infantry"]
    convert_actions = [a for a in legal if a["type"] == "convert_to_special"]

    chosen = []

    # Only one outpost action (build_outpost/upgrade_outpost, combined)
    # goes through per turn (see engine/buy.py's apply_buy_phase) -
    # upgrading an outpost already held is prioritized over building a
    # new one, since it's pure value-add on ground already secured.
    if upgrade_actions:
        by_hex = {}
        for a in upgrade_actions:
            by_hex.setdefault(a["hex"], {})[a["upgrade"]] = a
        upgrade_hexes = list(by_hex.keys())
        rng.shuffle(upgrade_hexes)
        options = by_hex[upgrade_hexes[0]]
        for upgrade in _UPGRADE_PRIORITY:
            if upgrade in options:
                chosen.append(options[upgrade])
                break
    else:
        # One preferred hex, shuffled among candidates, with a preferred
        # sacrificed unit_type (infantry first) at that hex.
        by_hex = {}
        for a in outpost_actions:
            by_hex.setdefault(a["hex"], {})[a["unit_type"]] = a
        outpost_hexes = list(by_hex.keys())
        rng.shuffle(outpost_hexes)
        if outpost_hexes:
            options = by_hex[outpost_hexes[0]]
            for unit_type in _OUTPOST_UNIT_PRIORITY:
                if unit_type in options:
                    chosen.append(options[unit_type])
                    break

    if infantry_actions:
        num_purchases = int(state.gold[faction]) // INFANTRY_COST
        for _ in range(num_purchases):
            chosen.append(rng.choice(infantry_actions))

    if convert_actions:
        num_conversions = int(state.kill_xp[faction])
        cav_count = count_units_in_play(state, faction, 1)
        arc_count = count_units_in_play(state, faction, 2)
        for _ in range(num_conversions):
            unit_type = "cavalry" if cav_count <= arc_count else "archers"
            matching = [a for a in convert_actions if a["unit_type"] == unit_type]
            if not matching:
                matching = convert_actions
            if matching:
                chosen.append(rng.choice(matching))

    return chosen


def _enemy_outpost_coords(state, faction):
    grid = state.grid
    idxs = np.nonzero((state.city_owner != NO_FACTION) & (state.city_owner != faction) & ~state.is_capital)[0]
    return [grid.coord_of(int(i)) for i in idxs]


def _enemy_capital_coords(state, faction):
    grid = state.grid
    idxs = np.nonzero(state.is_capital & (state.city_owner != NO_FACTION) & (state.city_owner != faction))[0]
    return [grid.coord_of(int(i)) for i in idxs]


def _home_expansion_target(state, faction):
    """The hex closest to `faction`'s own capital that's legal to found
    an outpost on right now, or None if this faction has already hit
    OUTPOST_CAP or nowhere on the board currently qualifies. Reuses
    buy.py's own outpost-placement legality check (eligible_outpost_mask)
    directly rather than re-deriving the same distance rules here, so
    a target this function suggests is always one greedy_buy could
    actually act on the moment an army gets there. Recomputed fresh on
    every call, fully vectorized (no per-hex Python loop) - once an
    outpost gets built at the current target, eligible_outpost_mask
    naturally stops accepting that hex and this just finds the
    next-nearest one, so expansion keeps going one spot at a time
    without any state to track between calls."""
    if _outpost_count(state, faction) >= OUTPOST_CAP:
        return None
    grid = state.grid
    own_capital = np.nonzero((state.city_owner == faction) & state.is_capital)[0]
    if len(own_capital) == 0:
        return None
    capital_index = int(own_capital[0])

    eligible = eligible_outpost_mask(state, faction) & ~IMPASSABLE_BY_TERRAIN[state.terrain]
    candidates = np.nonzero(eligible)[0]
    if len(candidates) == 0:
        return None
    dist = np.abs(grid.coords_array[candidates] - grid.coords_array[capital_index]).max(axis=1)
    best = int(candidates[np.argmin(dist)])
    return grid.coord_of(best)


def _move_toward(grid, ranked_origins, legal_mask, target_coords, skip_arrived=False):
    """Among ranked_origins (largest army first), returns the first
    (origin, direction) that takes a legal step toward the nearest of
    target_coords - no threat-awareness, always the closest-approaching
    legal direction regardless of what's in the way. With
    skip_arrived=True, an origin already sitting exactly on its nearest
    target is left alone instead of being nudged off it (used for the
    home-expansion target, so a settled army stays put for the buy
    phase to consume rather than wandering away); the attack-phase
    targets never need that since a hex is always in-battle rather than
    peacefully occupied the moment you're standing on an enemy outpost/
    capital, so "already arrived" can't happen there."""
    for origin in ranked_origins:
        origin_coord = grid.coord_of(origin)
        target = min(target_coords, key=lambda c: hex_distance(origin_coord, c))
        if skip_arrived and hex_distance(origin_coord, target) == 0:
            continue
        legal_dirs = np.nonzero(legal_mask[origin])[0]
        if len(legal_dirs) == 0:
            continue
        best_dir = min(
            legal_dirs,
            key=lambda d: hex_distance(grid.coord_of(int(grid.neighbor_table[origin, d])), target),
        )
        return origin, int(best_dir)
    return None


def greedy_rush_move(state, faction, legal_mask):
    """See module docstring for the two-priority (expand near home, then
    attack) strategy this implements."""
    grid = state.grid
    origins = np.nonzero(legal_mask.any(axis=1))[0]
    if len(origins) == 0:
        return None

    sizes = state.army_units[origins].sum(axis=1)
    # kind="stable": numpy's DEFAULT argsort is quicksort, whose order among
    # equal keys is an unreproducible implementation artifact (measured: it
    # differs from a stable sort in 62% of random 8-element size arrays, and
    # ~100% by 20 elements). Army sizes are small integers, so ties are the
    # common case, and which of two equally sized armies moves first was
    # therefore being decided by numpy internals. Pinning it to stable makes
    # the tie-break a plain property of the data - hex order - which the C++
    # port can reproduce. Same reasoning as engine/PLAN.md 3.2's sorted() fix
    # to terrain generation.
    ranked = [int(origins[i]) for i in np.argsort(-sizes, kind="stable")]

    home_target = _home_expansion_target(state, faction)
    if home_target is not None:
        move = _move_toward(grid, ranked, legal_mask, [home_target], skip_arrived=True)
        if move is not None:
            return move
        # every mobile army is already parked at the target (or none can
        # legally step toward it this turn) - fall through to attacking
        # instead of doing nothing while expansion is still available

    targets = _enemy_outpost_coords(state, faction) or _enemy_capital_coords(state, faction)
    if not targets:
        return None
    return _move_toward(grid, ranked, legal_mask, targets)


def _nearest_dist(grid, hex_index, other_hexes):
    """Hex distance from hex_index to the nearest hex in other_hexes."""
    coord = grid.coord_of(hex_index)
    return min(hex_distance(coord, grid.coord_of(h)) for h in other_hexes)


def greedy_placement(state, legal_mask):
    """Places on whichever legal hex is farthest from the nearest
    already-placed city - the same "farthest point" idea
    engine/setup.py used to apply automatically, now an agent choice."""
    grid = state.grid
    candidates = np.nonzero(legal_mask)[0].tolist()
    placed = np.nonzero(state.city_placer != NO_FACTION)[0].tolist()
    if not placed:
        return candidates[0]
    return max(candidates, key=lambda h: _nearest_dist(grid, h, placed))


def greedy_draft(state, legal_pool):
    """Drafts whichever legal pool city is farthest from the nearest
    already-claimed capital - a simple defensibility-flavored pick."""
    grid = state.grid
    claimed = np.nonzero(state.city_owner != NO_FACTION)[0].tolist()
    if not claimed:
        return legal_pool[0]
    return max(legal_pool, key=lambda h: _nearest_dist(grid, h, claimed))


def greedy_resource_choice(state, faction):
    """Prefers whichever of iron/fish this faction currently has less of -
    a simple balance heuristic, same spirit as greedy_buy's cavalry-vs-
    archers conversion choice."""
    iron = int(state.resources[faction, RESOURCE_TO_INDEX["iron"]])
    fish = int(state.resources[faction, RESOURCE_TO_INDEX["fish"]])
    return "iron" if iron <= fish else "fish"


def greedy_swap(state, leftover_hex, placer_hex):
    """Swaps only if placer_hex is a strict improvement over leftover_hex -
    farther from the nearest other already-claimed capital."""
    grid = state.grid
    others = [
        h for h in np.nonzero(state.city_owner != NO_FACTION)[0].tolist()
        if h not in (leftover_hex, placer_hex)
    ]
    if not others:
        return False
    return _nearest_dist(grid, placer_hex, others) > _nearest_dist(grid, leftover_hex, others)


def make_greedy_agents(num_factions, seed=0):
    """Returns (decide_buy, decide_movement, decide_cavalry, decide_target,
    decide_rectification, decide_placement, decide_draft, decide_swap) -
    each {faction: callable}, matching engine.turn.run_turn's and
    engine.placement.run_city_setup's expected signatures."""
    rngs = {f: random.Random(seed * 1_000_003 + f) for f in range(num_factions)}

    def decide_buy(state, faction, legal):
        return greedy_buy(state, faction, legal, rngs[faction])

    def decide_movement(state, faction, step, legal_mask):
        return greedy_rush_move(state, faction, legal_mask)

    def decide_cavalry(state, faction, step, legal_mask):
        return greedy_rush_move(state, faction, legal_mask)

    def decide_target(state, hex_index, faction):
        return random_target(state, hex_index, faction, rngs[faction])

    def decide_rectification(state, hex_index, winner_faction, cap):
        return random_rectification(state, hex_index, winner_faction, cap, rngs[winner_faction])

    def decide_resource_choice(state, faction, hex_index):
        return greedy_resource_choice(state, faction)

    def decide_placement(state, faction, legal_mask):
        return greedy_placement(state, legal_mask)

    def decide_draft(state, faction, legal_pool):
        return greedy_draft(state, legal_pool)

    def decide_swap(state, faction, leftover_hex, placer_faction, placer_hex):
        return greedy_swap(state, leftover_hex, placer_hex)

    factions = range(num_factions)
    return (
        {f: decide_buy for f in factions},
        {f: decide_movement for f in factions},
        {f: decide_cavalry for f in factions},
        {f: decide_target for f in factions},
        {f: decide_rectification for f in factions},
        {f: decide_resource_choice for f in factions},
        {f: decide_placement for f in factions},
        {f: decide_draft for f in factions},
        {f: decide_swap for f in factions},
    )
