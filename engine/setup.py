"""
Builds an initial ArrayState for a new game - ported from engine/setup.py's
create_initial_state. Same terrain-generation algorithm, just writing
directly into ArrayState's numpy arrays instead of a dict-of-HexState
board, so engine can create a game without going through v1 at all.

Capital placement is NOT done here - create_initial_state only generates
terrain and seeds starting silver/kill-XP, leaving city_owner/is_capital/
city_placer untouched (all NO_FACTION/False) for placement.py's
run_city_setup to fill in as a real agent-driven decision (colourless
placement, then a draft) rather than the farthest-point heuristic this
module used to apply automatically. See placement.py's module docstring
for that process, and turn.py's get_game_winner for the placement-order
tiebreak it unlocks.
"""

import random

import numpy as np

from .geometry import HexGrid
from .state import TERRAIN_TO_INDEX, new_empty

STARTING_SILVER = 50
STARTING_KILL_XP = 2

_UNSET = -1

# (min, max) hexes placed per round, inclusive - one entry per terrain type.
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

    edge_hexes = [i for i in unset if grid.is_edge(i)]
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

    for faction in range(num_factions):
        state.silver[faction] = STARTING_SILVER
        state.kill_xp[faction] = STARTING_KILL_XP

    return state
