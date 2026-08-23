"""
Builds an initial ArrayState for a new game - ported from engine/setup.py's
create_initial_state. Same algorithm (random terrain, farthest-point city
spread, 2-hex-away second city, starting silver/kill-XP), just writing
directly into ArrayState's numpy arrays instead of a dict-of-HexState
board, so engine can create a game without going through v1 at all.
"""

import random

import numpy as np

from .geometry import HexGrid, CUBE_DIRECTIONS, hex_distance
from .state import TERRAIN_TYPES, TERRAIN_TO_INDEX, NO_FACTION, new_empty

STARTING_SILVER = 50
STARTING_KILL_XP = 2

_IMPASSABLE = ("mountain", "lake")
_PASSABLE_TERRAIN = [t for t in TERRAIN_TYPES if t not in _IMPASSABLE]

_UNSET = -1

# (min, max) hexes placed per round, inclusive. Terrain types not listed
# here (forest) never come up in generation - forest is only reachable
# by hand-editing a board.
_ROUND_COUNTS = {
    "plains": (5, 8),
    "lake": (3, 5),
    "mountain": (2, 4),
    "marsh": (2, 5),
    "desert": (2, 5),
}

# Starting size of the terrain "bag": each round's type is drawn weighted
# by how many of that type are still left in the bag, and placing a hex
# removes one from it - so a type that comes up big early gets rarer for
# the rest of generation. Total (250) is kept comfortably above a
# radius-8 board's 217 hexes so the bag realistically never fully empties
# before the board does.
BAG_COUNTS = {
    "plains": 120,
    "lake": 25,
    "mountain": 25,
    "desert": 40,
    "marsh": 40,
}


def _same_type_neighbor_count(grid, terrain, index, type_index):
    count = 0
    for j in grid.neighbor_table[index]:
        if j != -1 and terrain[j] == type_index:
            count += 1
    return count


def _can_place(grid, terrain, index, type_name, type_index, placed_so_far):
    """Whether `index` may become the (placed_so_far + 1)-th hex placed
    this round, given the hexes of this type already on the board
    (including ones placed earlier this round)."""
    if type_name == "mountain":
        # First mountain of a round is free; every one after it must
        # extend the chain by exactly one link, keeping mountains linear.
        if placed_so_far == 0:
            return True
        return _same_type_neighbor_count(grid, terrain, index, type_index) == 1
    if type_name in ("lake", "marsh", "desert"):
        # First two hexes of a round are free; from the third on, the
        # hex must be growing an existing body rather than sprouting a
        # new one.
        if placed_so_far < 2:
            return True
        return _same_type_neighbor_count(grid, terrain, index, type_index) >= 2
    return True


def _place_round(grid, terrain, rng, start, type_name, round_index, log, bag):
    type_index = TERRAIN_TO_INDEX[type_name]
    lo, hi = _ROUND_COUNTS[type_name]
    target = min(rng.randint(lo, hi), bag[type_name])

    def place(index):
        terrain[index] = type_index
        if log is not None:
            q, r, s = grid.coord_of(index)
            log.append({"q": q, "r": r, "s": s, "terrain": type_name, "round": round_index})

    place(start)
    placed = [start]
    while len(placed) < target:
        candidates = set()
        for h in placed:
            for j in grid.neighbor_table[h]:
                if j != -1 and terrain[j] == _UNSET:
                    candidates.add(int(j))
        candidates = [
            c for c in candidates
            if _can_place(grid, terrain, c, type_name, type_index, len(placed))
        ]
        if not candidates:
            break
        choice = rng.choice(candidates)
        place(choice)
        placed.append(choice)
    bag[type_name] -= len(placed)
    return placed


def generate_terrain(grid, rng, log=None):
    """Builds a full terrain map in rounds: each round draws a hex type
    from a shrinking "bag" (weighted by how many of that type are left,
    see BAG_COUNTS) and grows a random-sized blob of it out from a spot
    touching the already-generated board (or, for the very first round,
    a random edge hex), subject to that type's placement rules.

    If `log` is given (a list), every individual hex placement is
    appended to it in placement order as {"q","r","s","terrain","round"}
    - see hex_gen.py for a step-by-step visualizer built on that log."""
    terrain = np.full(grid.num_hexes, _UNSET, dtype=np.int8)
    unset = set(range(grid.num_hexes))
    bag = dict(BAG_COUNTS)

    edge_hexes = [
        i for i in unset
        if max(abs(c) for c in grid.coords[i]) == grid.radius
    ]
    start = rng.choice(edge_hexes)

    round_index = 0
    while unset:
        types = [t for t, count in bag.items() if count > 0]
        weights = [bag[t] for t in types]
        type_name = rng.choices(types, weights=weights, k=1)[0]
        placed = _place_round(grid, terrain, rng, start, type_name, round_index, log, bag)
        round_index += 1
        unset.difference_update(placed)
        if unset:
            candidates = [
                i for i in unset
                if any(terrain[j] != _UNSET for j in grid.neighbor_table[i] if j != -1)
            ]
            start = rng.choice(candidates)

    return terrain


def create_initial_state(radius=8, num_factions=8, seed=42, terrain_log=None):
    """terrain_log: optional list - if given, receives every individual
    terrain-generation hex placement in order (see generate_terrain's
    docstring), for run.py to dump alongside board_state.json."""
    rng = random.Random(seed)
    grid = HexGrid(radius)
    state = new_empty(grid, num_factions)

    state.terrain[:] = generate_terrain(grid, rng, log=terrain_log)

    def ensure_passable(i):
        if TERRAIN_TYPES[state.terrain[i]] in _IMPASSABLE:
            state.terrain[i] = TERRAIN_TO_INDEX[rng.choice(_PASSABLE_TERRAIN)]

    all_indices = list(range(grid.num_hexes))
    home_indices = [rng.choice(all_indices)]
    for _ in range(num_factions - 1):
        best_index = max(
            all_indices,
            key=lambda i: min(
                hex_distance(grid.coord_of(i), grid.coord_of(h)) for h in home_indices
            ) + rng.random() * 0.01,
        )
        home_indices.append(best_index)

    for faction, home in enumerate(home_indices):
        ensure_passable(home)
        state.city_owner[home] = faction

        directions = list(CUBE_DIRECTIONS)
        rng.shuffle(directions)
        home_coord = grid.coord_of(home)
        second_city = None
        for dq, dr, ds in directions:
            cand = (home_coord[0] + 2 * dq, home_coord[1] + 2 * dr, home_coord[2] + 2 * ds)
            cand_index = grid.coord_to_index.get(cand)
            if cand_index is not None and state.city_owner[cand_index] == NO_FACTION:
                second_city = cand_index
                break
        if second_city is not None:
            ensure_passable(second_city)
            state.city_owner[second_city] = faction

        state.silver[faction] = STARTING_SILVER
        state.kill_xp[faction] = STARTING_KILL_XP

    return state
