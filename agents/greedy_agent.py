"""
GreedyAgent for engine: a single-minded outpost rush.

Buy phase prioritizes its one outpost action for the turn: upgrading an
outpost it already holds (temple > barracks > workshop - direct VP
first, then whatever compounds economy/resources) comes before building
a new one (sacrificing infantry > cavalry > archers, in that priority).
Whatever gold is left then gets dumped on infantry at one city hex, and
all banked kill-XP gets converted toward whichever of cavalry/archers it
currently has fewer of.

BUY-PHASE REDESIGN (see engine/buy.py's module docstring): the old
per-atomic-action list this used to filter (get_legal_buy_actions) is
gone - buy.py's fixed-shape action space replaced it. greedy_buy now
inspects `state` directly and returns that fixed-shape decision dict.
The actual PRIORITIES (upgrade-over-build, infantry dumped at one hex,
kill-XP alternating toward whichever of cavalry/archers is behind) are
unchanged from before the rewrite - only how a decision gets expressed
changed, not what greedy_agent actually wants to do.

Movement/cavalry phases have two priorities, strictly ordered: while this
faction still has room for more outposts (under OUTPOST_CAP) and at
least one hex anywhere on the board is currently legal to found one on,
every decision funnels armies toward the single such hex closest to this
faction's own capital - once one army settles there the buy phase builds
on it (see greedy_buy), and the next call just finds the new nearest
spot, so expansion keeps going one outpost at a time rather than
scattering everywhere at once. Only once that's exhausted (cap reached,
or literally nowhere legal left) do armies switch to attacking: straight
at the nearest enemy outpost to destroy it, or, if none exist yet, the
nearest enemy capital instead, just to keep units advancing rather than
sitting idle.

No retreat logic, no threat-aware pathing around stronger armies en
route - always takes whichever legal step most closes distance to the
current target, for better or worse. That single-mindedness is what
"greedy" means here, not tactical caution.

decide_target/decide_rectification are reused as-is from random_agent.py
- neither needs to be smart for this agent's strategy to work.
decide_resource_choice gets a one-line heuristic (greedy_resource_choice:
prefer whichever of iron/fish is scarcer) rather than being reused as-is.
Setup-phase decisions (placement/draft/swap) get their own greedy policy
below: farthest-point placement/drafting.
"""

import random

import torch

from .random_agent import random_rectification, random_target
from engine.buy import OUTPOST_CAP, UPGRADE_COSTS, eligible_outpost_mask, _outpost_count
from engine.geometry import hex_distance
from engine.state import (
    IMPASSABLE_BY_TERRAIN, NO_FACTION, NO_UPGRADE, RESOURCE_TO_INDEX, UPGRADE_TO_INDEX, count_units_in_play,
)

# Which unit type to sacrifice first when a hex offers a choice of more
# than one for building an outpost there.
_OUTPOST_UNIT_PRIORITY = (0, 1, 2)  # infantry, cavalry, archers

# Which upgrade to grab first when more than one is affordable: Temple
# first (direct VP - the actual win condition), then Barracks (compounds
# economy), then Workshop (compounds resources) - a "greedy" ranking by
# how directly each pays off.
_UPGRADE_PRIORITY = (UPGRADE_TO_INDEX["temple"], UPGRADE_TO_INDEX["barracks"], UPGRADE_TO_INDEX["workshop"])


def _can_afford_upgrade(state, faction, upgrade_index):
    cost = UPGRADE_COSTS[upgrade_index]
    return all(int(state.resources[0, faction, RESOURCE_TO_INDEX[r]]) >= amount for r, amount in cost.items())


def greedy_buy(state, faction, rng):
    decision = {}

    outposts = torch.nonzero(
        (state.city_owner[0] == faction) & ~state.is_capital[0] & ~state.locked[0], as_tuple=False
    ).flatten().tolist()
    upgradeable = [
        h for h in outposts
        if int(state.outpost_upgrade[0, h]) == NO_UPGRADE and any(_can_afford_upgrade(state, faction, u) for u in _UPGRADE_PRIORITY)
    ]

    if upgradeable:
        h = rng.choice(upgradeable)
        for upgrade in _UPGRADE_PRIORITY:
            if _can_afford_upgrade(state, faction, upgrade):
                decision["outpost_type"] = 2
                decision["outpost_hex"] = h
                decision["outpost_upgrade"] = upgrade
                break
    else:
        army_hexes = torch.nonzero(
            (state.army_faction[0] == faction) & ~state.locked[0], as_tuple=False
        ).flatten().tolist()
        if army_hexes and int(state.gold[0, faction]) >= 3 and int(_outpost_count(state, faction)[0]) < OUTPOST_CAP:
            eligible = eligible_outpost_mask(state, faction)[0]
            candidates = [h for h in army_hexes if bool(eligible[h])]
            rng.shuffle(candidates)
            for h in candidates:
                for unit_type in _OUTPOST_UNIT_PRIORITY:
                    if int(state.army_units[0, h, unit_type]) > 0:
                        decision["outpost_type"] = 1
                        decision["outpost_hex"] = h
                        decision["outpost_unit_type"] = unit_type
                        break
                if "outpost_type" in decision:
                    break

    city_hexes = torch.nonzero((state.city_owner[0] == faction) & ~state.locked[0], as_tuple=False).flatten().tolist()
    if city_hexes and int(state.gold[0, faction]) >= 2:
        h = rng.choice(city_hexes)
        decision["infantry_buy"] = {h: int(state.gold[0, faction]) // 2}

    army_hexes = torch.nonzero(
        (state.army_faction[0] == faction) & (state.army_units[0, :, 0] > 0), as_tuple=False
    ).flatten().tolist()
    kill_xp = int(state.kill_xp[0, faction])
    if army_hexes and kill_xp > 0:
        cav_count = int(count_units_in_play(state, faction, 1)[0])
        arc_count = int(count_units_in_play(state, faction, 2)[0])
        convert_cavalry, convert_archers = {}, {}
        for _ in range(kill_xp):
            h = rng.choice(army_hexes)
            if cav_count <= arc_count:
                convert_cavalry[h] = convert_cavalry.get(h, 0) + 1
                cav_count += 1
            else:
                convert_archers[h] = convert_archers.get(h, 0) + 1
                arc_count += 1
        if convert_cavalry:
            decision["convert_cavalry"] = convert_cavalry
        if convert_archers:
            decision["convert_archers"] = convert_archers

    return decision


def _enemy_outpost_coords(state, faction):
    grid = state.grid
    idxs = torch.nonzero(
        (state.city_owner[0] != NO_FACTION) & (state.city_owner[0] != faction) & ~state.is_capital[0], as_tuple=False
    ).flatten().tolist()
    return [grid.coord_of(i) for i in idxs]


def _enemy_capital_coords(state, faction):
    grid = state.grid
    idxs = torch.nonzero(
        state.is_capital[0] & (state.city_owner[0] != NO_FACTION) & (state.city_owner[0] != faction), as_tuple=False
    ).flatten().tolist()
    return [grid.coord_of(i) for i in idxs]


def _home_expansion_target(state, faction):
    """The hex closest to `faction`'s own capital that's legal to found
    an outpost on right now, or None if this faction has already hit
    OUTPOST_CAP or nowhere on the board currently qualifies. Reuses
    buy.py's own outpost-placement legality check (eligible_outpost_mask)
    directly rather than re-deriving the same distance rules here, so a
    target this function suggests is always one greedy_buy could
    actually act on the moment an army gets there. Recomputed fresh on
    every call, fully vectorized (no per-hex Python loop)."""
    if int(_outpost_count(state, faction)[0]) >= OUTPOST_CAP:
        return None
    grid = state.grid
    own_capital = torch.nonzero((state.city_owner[0] == faction) & state.is_capital[0], as_tuple=False).flatten()
    if len(own_capital) == 0:
        return None
    capital_index = int(own_capital[0])

    eligible = eligible_outpost_mask(state, faction)[0] & ~IMPASSABLE_BY_TERRAIN.to(state.device)[state.terrain[0].long()]
    candidates = torch.nonzero(eligible, as_tuple=False).flatten()
    if len(candidates) == 0:
        return None
    dist = (grid.coords_array[candidates] - grid.coords_array[capital_index]).abs().amax(dim=-1)
    best = int(candidates[torch.argmin(dist)])
    return grid.coord_of(best)


def _move_toward(grid, ranked_origins, legal_mask, target_coords, skip_arrived=False):
    """Among ranked_origins (largest army first), returns the first
    (origin, direction) that takes a legal step toward the nearest of
    target_coords - no threat-awareness, always the closest-approaching
    legal direction regardless of what's in the way. With
    skip_arrived=True, an origin already sitting exactly on its nearest
    target is left alone instead of being nudged off it."""
    for origin in ranked_origins:
        origin_coord = grid.coord_of(origin)
        target = min(target_coords, key=lambda c: hex_distance(origin_coord, c))
        if skip_arrived and hex_distance(origin_coord, target) == 0:
            continue
        legal_dirs = torch.nonzero(legal_mask[origin], as_tuple=False).flatten().tolist()
        if not legal_dirs:
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
    origins = torch.nonzero(legal_mask.any(dim=1), as_tuple=False).flatten()
    if len(origins) == 0:
        return None

    sizes = state.army_units[0, origins].sum(dim=1)
    ranked = [int(origins[i]) for i in torch.argsort(-sizes)]

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
    already-placed city - the same "farthest point" idea engine/setup.py
    used to apply automatically, now an agent choice."""
    grid = state.grid
    candidates = torch.nonzero(legal_mask, as_tuple=False).flatten().tolist()
    placed = torch.nonzero(state.city_placer[0] != NO_FACTION, as_tuple=False).flatten().tolist()
    if not placed:
        return candidates[0]
    return max(candidates, key=lambda h: _nearest_dist(grid, h, placed))


def greedy_draft(state, legal_pool):
    """Drafts whichever legal pool city is farthest from the nearest
    already-claimed capital - a simple defensibility-flavored pick."""
    grid = state.grid
    claimed = torch.nonzero(state.city_owner[0] != NO_FACTION, as_tuple=False).flatten().tolist()
    if not claimed:
        return legal_pool[0]
    return max(legal_pool, key=lambda h: _nearest_dist(grid, h, claimed))


def greedy_resource_choice(state, faction):
    """Prefers whichever of iron/fish this faction currently has less of -
    a simple balance heuristic, same spirit as greedy_buy's cavalry-vs-
    archers conversion choice."""
    iron = int(state.resources[0, faction, RESOURCE_TO_INDEX["iron"]])
    fish = int(state.resources[0, faction, RESOURCE_TO_INDEX["fish"]])
    return "iron" if iron <= fish else "fish"


def greedy_swap(state, leftover_hex, placer_hex):
    """Swaps only if placer_hex is a strict improvement over leftover_hex -
    farther from the nearest other already-claimed capital."""
    grid = state.grid
    others = [
        h for h in torch.nonzero(state.city_owner[0] != NO_FACTION, as_tuple=False).flatten().tolist()
        if h not in (leftover_hex, placer_hex)
    ]
    if not others:
        return False
    return _nearest_dist(grid, placer_hex, others) > _nearest_dist(grid, leftover_hex, others)


def make_greedy_agents(num_factions, seed=0):
    """Returns (decide_buy, decide_movement, decide_cavalry, decide_target,
    decide_rectification, decide_resource_choice, decide_placement,
    decide_draft, decide_swap) - each {faction: callable}, matching
    engine.turn.run_turn's and engine.placement.run_city_setup's expected
    signatures."""
    rngs = {f: random.Random(seed * 1_000_003 + f) for f in range(num_factions)}

    def decide_buy(state, faction):
        return greedy_buy(state, faction, rngs[faction])

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
