"""
GreedyAgent for engine: a single-minded outpost rush.

Buy phase prioritizes building as many outposts as it can afford and has
units for, then spends whatever's left the same aggressive way as
heuristic_agent/smart_random_agent - infantry with leftover silver, all
banked kill-XP converted toward whichever of cavalry/archers it currently
has fewer of.

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

Deliberately blunter than heuristic_agent: no retreat logic, no threat-
aware pathing around stronger armies en route - always takes whichever
legal step most closes distance to the current target, for better or
worse. That single-mindedness is what "greedy" means here, not tactical
caution.

decide_target/decide_rectification are reused as-is from random_agent.py
(neither needs to be smart for this agent's strategy to work, same as
every other scripted agent in this package), and the setup-phase
decisions (placement/draft/swap) aren't part of this agent's stated
strategy either, so those are borrowed from smart_random_agent.py's
farthest-point-greedy policies instead of plain random - a better
thematic fit for "greedy" than a uniform coin flip.
"""

import random

import numpy as np

from .random_agent import random_rectification, random_target
from .smart_random_agent import smart_draft, smart_placement, smart_swap
from engine.buy import INFANTRY_COST, OUTPOST_CAP, _can_build_outpost, _outpost_count
from engine.geometry import hex_distance
from engine.state import IMPASSABLE_TERRAIN_INDICES, NO_FACTION, count_units_in_play

# Which unit type to sacrifice first when a hex offers a choice of more
# than one for building an outpost there.
_OUTPOST_UNIT_PRIORITY = ("infantry", "cavalry", "archers")

_IMPASSABLE_INDEX_SET = {int(x) for x in IMPASSABLE_TERRAIN_INDICES}


def greedy_buy(state, faction, legal, rng):
    outpost_actions = [a for a in legal if a["type"] == "build_outpost"]
    infantry_actions = [a for a in legal if a["type"] == "buy_infantry"]
    convert_actions = [a for a in legal if a["type"] == "convert_to_special"]

    chosen = []

    # One outpost attempt per hex: get_legal_buy_actions returns one
    # build_outpost entry per (hex, unit_type present), but only the
    # first attempt at a given hex can ever succeed (building sets
    # city_owner there, which _can_build_outpost then rejects) - picking
    # one preferred entry per hex up front, rather than submitting every
    # entry and relying on that silent-drop behavior, keeps which unit
    # type gets sacrificed a deliberate choice instead of an accident of
    # list order.
    by_hex = {}
    for a in outpost_actions:
        by_hex.setdefault(a["hex"], {})[a["unit_type"]] = a
    outpost_hexes = list(by_hex.keys())
    rng.shuffle(outpost_hexes)
    for hex_index in outpost_hexes:
        options = by_hex[hex_index]
        for unit_type in _OUTPOST_UNIT_PRIORITY:
            if unit_type in options:
                chosen.append(options[unit_type])
                break

    if infantry_actions:
        num_purchases = int(state.silver[faction]) // INFANTRY_COST
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
    buy.py's own outpost-placement legality check (_can_build_outpost)
    directly rather than re-deriving the same distance rules here, so
    a target this function suggests is always one greedy_buy could
    actually act on the moment an army gets there. Recomputed fresh on
    every call - once an outpost gets built at the current target,
    _can_build_outpost naturally stops accepting that hex and this
    just finds the next-nearest one, so expansion keeps going one spot
    at a time without any state to track between calls."""
    if _outpost_count(state, faction) >= OUTPOST_CAP:
        return None
    grid = state.grid
    own_capital = np.nonzero((state.city_owner == faction) & state.is_capital)[0]
    if len(own_capital) == 0:
        return None
    capital_coord = grid.coord_of(int(own_capital[0]))

    eligible = [
        i for i in range(grid.num_hexes)
        if int(state.terrain[i]) not in _IMPASSABLE_INDEX_SET and _can_build_outpost(state, i, faction)
    ]
    if not eligible:
        return None
    best = min(eligible, key=lambda i: hex_distance(grid.coord_of(i), capital_coord))
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
    ranked = [int(origins[i]) for i in np.argsort(-sizes)]

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

    def decide_placement(state, faction, legal_mask):
        return smart_placement(state, legal_mask)

    def decide_draft(state, faction, legal_pool):
        return smart_draft(state, legal_pool)

    def decide_swap(state, faction, leftover_hex, placer_faction, placer_hex):
        return smart_swap(state, leftover_hex, placer_hex)

    factions = range(num_factions)
    return (
        {f: decide_buy for f in factions},
        {f: decide_movement for f in factions},
        {f: decide_cavalry for f in factions},
        {f: decide_target for f in factions},
        {f: decide_rectification for f in factions},
        {f: decide_placement for f in factions},
        {f: decide_draft for f in factions},
        {f: decide_swap for f in factions},
    )
