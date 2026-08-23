"""
Builds an initial ArrayState for a new game - ported from engine/setup.py's
create_initial_state. Same algorithm (random terrain, farthest-point city
spread, 2-hex-away second city, starting silver/kill-XP), just writing
directly into ArrayState's numpy arrays instead of a dict-of-HexState
board, so engine_v2 can create a game without going through v1 at all.
"""

import random

from .geometry import HexGrid, CUBE_DIRECTIONS, hex_distance
from .state import TERRAIN_TYPES, TERRAIN_TO_INDEX, NO_FACTION, new_empty

STARTING_SILVER = 50
STARTING_KILL_XP = 2

_IMPASSABLE = ("mountain", "lake")
_PASSABLE_TERRAIN = [t for t in TERRAIN_TYPES if t not in _IMPASSABLE]


def create_initial_state(radius=8, num_factions=8, seed=42):
    rng = random.Random(seed)
    grid = HexGrid(radius)
    state = new_empty(grid, num_factions)

    for i in range(grid.num_hexes):
        state.terrain[i] = TERRAIN_TO_INDEX[rng.choice(TERRAIN_TYPES)]

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
