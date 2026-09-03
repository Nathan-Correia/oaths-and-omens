"""
HeuristicAgent for engine: reuses greedy_agent's buy phase, rectification,
and setup-phase (placement/draft/swap/resource-choice) policies as-is,
and replaces only the two places greedy is purely myopic - battle
targeting and where armies go - with threat-aware versions:

  - decide_target (heuristic_target) attacks whichever rival in the fight
    currently has the FEWEST total units, not a uniform-random one -
    finishing off the weakest contributor first shrinks the number of
    rolls landing on you every subsequent round, the "kill the squishy
    target first" logic from any RTS.

  - Attack-phase movement (see _best_attack_target) scores every enemy
    outpost by estimated defender power (current garrison + its 1 free
    structure-defense shot) instead of nearest-only, and attacks
    whichever is weakest-defended.

  - Home-expansion site scoring (see _score_expansion_site) extends
    greedy's pure nearest-to-capital metric with a resource tie-break
    (bonus for mountain/lake adjacency; penalty for a desert with
    neither, which produces nothing) among every site within
    EXPANSION_TOLERANCE hexes of the single nearest one.
"""

import random

import torch

from .greedy_agent import _move_toward, greedy_buy, greedy_draft, greedy_placement, greedy_resource_choice, greedy_swap
from .random_agent import random_rectification
from engine.battle import faction_totals, get_legal_target_actions
from engine.buy import OUTPOST_CAP, eligible_outpost_mask, _outpost_count
from engine.geometry import hex_distance
from engine.state import IMPASSABLE_BY_TERRAIN, NO_FACTION, TERRAIN_TO_INDEX

MOUNTAIN_INDEX = TERRAIN_TO_INDEX["mountain"]
LAKE_INDEX = TERRAIN_TO_INDEX["lake"]
DESERT_INDEX = TERRAIN_TO_INDEX["desert"]

# Rough relative combat value per unit type - archers/cavalry outvalue a
# plain infantry unit thanks to their battle abilities. Only used to
# compare stack strength for threat/target scoring, never for the actual
# battle math (which the engine resolves exactly - see engine/battle.py).
_UNIT_POWER = torch.tensor([1.0, 1.2, 1.3])

EXPANSION_TOLERANCE = 2       # hexes of extra distance a richer expansion site is allowed to cost
                              # over the single nearest one - see _score_expansion_site's docstring
ATTACK_TOLERANCE = 2         # same idea for _best_attack_target: how many extra hexes a weaker-
                              # defended outpost is allowed to cost over the single nearest one
OUTPOST_DEFENSE_POWER = 0.5  # ~expected kills from an outpost's 1 free defense shot (11-20 on a d20)


def _power(units):
    return float((units.float() * _UNIT_POWER.to(units.device)).sum())


def heuristic_target(state, hex_index, faction):
    """Attacks whichever rival currently in the fight has the fewest
    total units - see module docstring."""
    legal = get_legal_target_actions(state, hex_index, faction)
    if not legal:
        return None
    totals = faction_totals(state, hex_index)
    return min(legal, key=lambda f: int(totals[f].sum()))


def _resource_bonus(state, hex_index):
    """+1 per mountain/lake neighbor the outpost would harvest (capped at
    2 total - two neighbors just means a genuine choice, not double
    value), or a penalty if this hex is a desert with neither (produces
    nothing at all)."""
    grid = state.grid
    neighbors = grid.neighbor_table[hex_index]
    terrain = state.terrain[0]
    has_mountain = any(int(j) != -1 and int(terrain[j]) == MOUNTAIN_INDEX for j in neighbors)
    has_lake = any(int(j) != -1 and int(terrain[j]) == LAKE_INDEX for j in neighbors)
    if has_mountain and has_lake:
        return 2.0
    if has_mountain or has_lake:
        return 1.0
    if int(terrain[hex_index]) == DESERT_INDEX:
        return -2.0
    return 0.0


def _score_expansion_site(state, hex_index, distance):
    """Distance (dominant - see _best_expansion_target) plus a resource
    tie-break. Deliberately NOT a function of nearby enemy army strength
    - see this repo's history for why a "danger" term thrashed target
    selection turn to turn without any real change in position; distance
    and terrain are both static, so scoring on those alone is stable by
    construction."""
    resource = _resource_bonus(state, hex_index)
    return -distance + resource


def _best_expansion_target(state, faction):
    """Like greedy_agent's _home_expansion_target, but breaks ties among
    every site within EXPANSION_TOLERANCE hexes of the single nearest one
    by resources (see _score_expansion_site) instead of always taking the
    strictly-nearest hex."""
    if int(_outpost_count(state, faction)[0]) >= OUTPOST_CAP:
        return None
    grid = state.grid
    own_capital = torch.nonzero((state.city_owner[0] == faction) & state.is_capital[0], as_tuple=False).flatten()
    if len(own_capital) == 0:
        return None
    capital_index = int(own_capital[0])

    eligible = eligible_outpost_mask(state, faction)[0] & ~IMPASSABLE_BY_TERRAIN.to(state.device)[state.terrain[0].long()]
    candidates = torch.nonzero(eligible, as_tuple=False).flatten().tolist()
    if not candidates:
        return None
    dist_arr = (grid.coords_array[candidates] - grid.coords_array[capital_index]).abs().amax(dim=-1)
    distances = {h: int(d) for h, d in zip(candidates, dist_arr.tolist())}
    min_dist = min(distances.values())
    tolerant = [h for h in candidates if distances[h] <= min_dist + EXPANSION_TOLERANCE]
    best = max(tolerant, key=lambda h: _score_expansion_site(state, h, distances[h]))
    return grid.coord_of(best)


def _best_attack_target(state, faction):
    """Nearest enemy outpost to our biggest mobile army, tie-broken
    (within ATTACK_TOLERANCE hexes of that nearest one) by weakest-
    defended. Falls back to every enemy capital (nearest wins, via
    _move_toward) if no enemy outpost exists yet."""
    grid = state.grid
    origins = torch.nonzero((state.army_faction[0] == faction) & ~state.locked[0], as_tuple=False).flatten()
    if len(origins) == 0:
        return None
    sizes = state.army_units[0, origins].sum(dim=1)
    ref_coord = grid.coord_of(int(origins[int(torch.argmax(sizes))]))

    outposts = torch.nonzero(
        (state.city_owner[0] != NO_FACTION) & (state.city_owner[0] != faction) & ~state.is_capital[0], as_tuple=False
    ).flatten().tolist()
    if outposts:
        distances = {o: hex_distance(ref_coord, grid.coord_of(o)) for o in outposts}
        min_dist = min(distances.values())
        candidates = [o for o, d in distances.items() if d <= min_dist + ATTACK_TOLERANCE]
        best = min(candidates, key=lambda o: _power(state.army_units[0, o]) + OUTPOST_DEFENSE_POWER)
        return [grid.coord_of(best)]

    capitals = torch.nonzero(
        state.is_capital[0] & (state.city_owner[0] != NO_FACTION) & (state.city_owner[0] != faction), as_tuple=False
    ).flatten().tolist()
    if not capitals:
        return None
    return [grid.coord_of(c) for c in capitals]


def heuristic_move(state, faction, legal_mask):
    grid = state.grid
    origins = torch.nonzero(legal_mask.any(dim=1), as_tuple=False).flatten()
    if len(origins) == 0:
        return None

    sizes = state.army_units[0, origins].sum(dim=1)
    ranked = [int(origins[i]) for i in torch.argsort(-sizes)]

    home_target = _best_expansion_target(state, faction)
    if home_target is not None:
        move = _move_toward(grid, ranked, legal_mask, [home_target], skip_arrived=True)
        if move is not None:
            return move
        # every mobile army already parked at the target (or none can
        # legally step toward it) - fall through to attacking instead of
        # doing nothing while expansion is still theoretically available

    attack_targets = _best_attack_target(state, faction)
    if attack_targets is None:
        return None
    return _move_toward(grid, ranked, legal_mask, attack_targets)


def make_heuristic_agents(num_factions, seed=0):
    """Same 9-callback shape as make_greedy_agents/make_random_agents.
    Buy/rectification/resource-choice/placement/draft/swap are greedy's
    as-is; movement and battle targeting are this module's."""
    rngs = {f: random.Random(seed * 1_000_003 + f) for f in range(num_factions)}

    def decide_buy(state, faction):
        return greedy_buy(state, faction, rngs[faction])

    def decide_movement(state, faction, step, legal_mask):
        return heuristic_move(state, faction, legal_mask)

    def decide_cavalry(state, faction, step, legal_mask):
        return heuristic_move(state, faction, legal_mask)

    def decide_target(state, hex_index, faction):
        return heuristic_target(state, hex_index, faction)

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
