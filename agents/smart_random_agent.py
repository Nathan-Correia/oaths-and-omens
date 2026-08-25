"""
SmartRandomAgent for engine - ported from
engine/agents/smart_random_agent.py. A step up from pure random in
exactly the same two ways as v1:

  1. Buy phase: spends all available silver on infantry, and converts
     every banked kill-XP token into a cavalry or archer unit (type
     chosen randomly per token).
  2. Movement / cavalry phases: greedily moves its single largest
     eligible army one step toward whichever enemy city is currently
     nearest to it (hex distance). engine's decide_movement already
     only gets to return ONE action per faction per step (see
     movement.py's SCOPE decision), so "pick the biggest army" is just
     which origin this function ranks first, not a constraint it has to
     enforce separately the way v1's agent did against a list-shaped
     legal-action set.

Battle targeting and rectification are reused unchanged from
random_agent.py, matching v1's inheritance chain.

  3. Setup phase (see engine/placement.py): places its colourless city on
     whichever legal hex is farthest from the nearest already-placed
     city (the same "farthest point" idea engine/setup.py used to apply
     automatically, now expressed as an agent choice), drafts whichever
     legal pool city is farthest from the nearest already-claimed
     capital (a simple defensibility-flavored greedy pick), and swaps
     only when that's a strict improvement over the leftover. All three
     are single-shot greedy picks, same "greedy, not globally optimal"
     spirit as greedy_move_largest_army above.
"""

import random

import numpy as np

from .random_agent import random_rectification, random_target
from engine.buy import INFANTRY_COST
from engine.geometry import hex_distance
from engine.state import NO_FACTION


def _enemy_city_coords(state, faction):
    grid = state.grid
    idxs = np.nonzero((state.city_owner != NO_FACTION) & (state.city_owner != faction))[0]
    return [grid.coord_of(int(i)) for i in idxs]


def greedy_move_largest_army(state, faction, legal_mask):
    """For every origin with a legal move this step, ranked largest-army
    first, takes the single step (among that origin's legal directions)
    that most reduces hex distance to the nearest enemy city. Mirrors
    v1's _greedy_move_largest_army."""
    grid = state.grid
    origins = np.nonzero(legal_mask.any(axis=1))[0]
    if len(origins) == 0:
        return None

    enemy_cities = _enemy_city_coords(state, faction)
    if not enemy_cities:
        return None

    sizes = state.army_units[origins].sum(axis=1)
    ranked = [int(origins[i]) for i in np.argsort(-sizes)]

    for origin in ranked:
        target_city = min(enemy_cities, key=lambda c: hex_distance(grid.coord_of(origin), c))
        legal_dirs = np.nonzero(legal_mask[origin])[0]
        if len(legal_dirs) == 0:
            continue
        best_dir = min(
            legal_dirs,
            key=lambda d: hex_distance(grid.coord_of(int(grid.neighbor_table[origin, d])), target_city),
        )
        return origin, int(best_dir)

    return None


def smart_buy(state, faction, legal, rng):
    infantry_actions = [a for a in legal if a["type"] == "buy_infantry"]
    convert_actions = [a for a in legal if a["type"] == "convert_to_special"]

    chosen = []

    if infantry_actions:
        num_purchases = int(state.silver[faction]) // INFANTRY_COST
        for _ in range(num_purchases):
            chosen.append(rng.choice(infantry_actions))

    if convert_actions:
        num_conversions = int(state.kill_xp[faction])
        for _ in range(num_conversions):
            unit_type = rng.choice(["cavalry", "archers"])
            matching = [a for a in convert_actions if a["unit_type"] == unit_type]
            if matching:
                chosen.append(rng.choice(matching))

    return chosen


def _nearest_dist(grid, hex_index, other_hexes):
    """Hex distance from hex_index to the nearest hex in other_hexes."""
    coord = grid.coord_of(hex_index)
    return min(hex_distance(coord, grid.coord_of(h)) for h in other_hexes)


def smart_placement(state, legal_mask):
    grid = state.grid
    candidates = np.nonzero(legal_mask)[0].tolist()
    placed = np.nonzero(state.city_placer != NO_FACTION)[0].tolist()
    if not placed:
        return candidates[0]
    return max(candidates, key=lambda h: _nearest_dist(grid, h, placed))


def smart_draft(state, legal_pool):
    grid = state.grid
    claimed = np.nonzero(state.city_owner != NO_FACTION)[0].tolist()
    if not claimed:
        return legal_pool[0]
    return max(legal_pool, key=lambda h: _nearest_dist(grid, h, claimed))


def smart_swap(state, leftover_hex, placer_hex):
    grid = state.grid
    others = [
        h for h in np.nonzero(state.city_owner != NO_FACTION)[0].tolist()
        if h not in (leftover_hex, placer_hex)
    ]
    if not others:
        return False
    return _nearest_dist(grid, placer_hex, others) > _nearest_dist(grid, leftover_hex, others)


def make_smart_random_agents(num_factions, seed=0):
    """Returns (decide_buy, decide_movement, decide_cavalry, decide_target,
    decide_rectification, decide_placement, decide_draft, decide_swap) -
    each {faction: callable}, matching engine.turn.run_turn's and
    engine.placement.run_city_setup's expected signatures."""
    rngs = {f: random.Random(seed * 1_000_003 + f) for f in range(num_factions)}

    def decide_buy(state, faction, legal):
        return smart_buy(state, faction, legal, rngs[faction])

    def decide_movement(state, faction, step, legal_mask):
        return greedy_move_largest_army(state, faction, legal_mask)

    def decide_cavalry(state, faction, step, legal_mask):
        return greedy_move_largest_army(state, faction, legal_mask)

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
