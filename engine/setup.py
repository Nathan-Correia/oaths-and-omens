"""
Builds an initial ArrayState for a new game (batch_size=1 - see module
docstring's note below on why this isn't batched).

Capital placement is NOT done here - create_initial_state only generates
terrain and seeds starting gold/kill-XP, leaving city_owner/is_capital/
city_placer untouched (all NO_FACTION/False) for placement.py's
run_city_setup to fill in as a real agent-driven decision.

NOT BATCHED, DELIBERATELY (see the plan's scope section): this is a
strictly sequential, order-dependent algorithm (a shrinking "bag," per-
round retry logic, a stuck-rounds circuit breaker, a full-board
connectivity check on many candidate placements) with no vectorizable
core, and it's a one-time per-game cost, not a per-turn one - the actual
GPU-batching payoff is in the repeating turn loop (engine/turn.py), not
here. To build a batch of B games, call create_initial_state B times
(each producing a batch_size=1 ArrayState) and stack_states() them - see
state.py.

The terrain-generation algorithm itself works on a plain Python list, not
a torch tensor - HexGrid.neighbor_table is converted to a list once up
front (_neighbor_lists) so this sequential, hex-by-hex bookkeeping reads
and iterates like ordinary Python rather than fighting torch's 0-d-tensor
iteration ergonomics; only the FINAL terrain array is written into the
(torch-backed) ArrayState.
"""

import random

import torch

from .geometry import HexGrid
from .state import IMPASSABLE_TERRAIN_INDICES, TERRAIN_TO_INDEX, new_empty

STARTING_GOLD = 50
STARTING_KILL_XP = 2

_UNSET = -1
_IMPASSABLE_TYPES = ("mountain", "lake")
_IMPASSABLE_INDEX_SET = {int(x) for x in IMPASSABLE_TERRAIN_INDICES}

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


def _neighbor_lists(grid):
    """[[neighbor_index_or_-1, ...] * 6, ...] - grid.neighbor_table as
    plain Python lists, once, for the sequential terrain-gen algorithm
    below to iterate without torch tensor overhead."""
    return grid.neighbor_table.tolist()


def _same_type_neighbor_count(neighbors, terrain, index, type_index):
    count = 0
    for j in neighbors[index]:
        if j != -1 and terrain[j] == type_index:
            count += 1
    return count


def _would_disconnect(grid, neighbors, terrain, candidate):
    """True if marking hex `candidate` impassable would split the rest of
    the board's non-impassable hexes into more than one connected
    component - i.e. would wall part of the board off into an
    unreachable island. Unset hexes count as non-impassable here too:
    every unset hex is guaranteed to end up as some terrain type before
    generation finishes, and "impassable" only ever means mountain/lake,
    so "currently non-impassable" is exactly "will end up passable,
    eventually." A plain BFS/DFS over a ~150-300 hex board is cheap
    enough to just rerun from scratch each time this is called - this
    only runs during one-off map generation, never in the per-turn hot
    path."""
    passable = [
        i for i in range(grid.num_hexes)
        if i != candidate and terrain[i] not in _IMPASSABLE_INDEX_SET
    ]
    if not passable:
        return False
    passable_set = set(passable)
    seen = {passable[0]}
    stack = [passable[0]]
    while stack:
        h = stack.pop()
        for j in neighbors[h]:
            if j != -1 and j in passable_set and j not in seen:
                seen.add(j)
                stack.append(j)
    return len(seen) != len(passable_set)


def _can_place(grid, neighbors, terrain, index, type_name, type_index, placed_so_far, check_disconnect=True):
    """Whether `index` may become the (placed_so_far + 1)-th hex placed
    this round, given the hexes of this type already on the board
    (including ones placed earlier this round)."""
    if type_name == "mountain":
        # First mountain of a round is free; every one after it must
        # extend the chain by exactly one link, keeping mountains linear.
        if placed_so_far == 0:
            shape_ok = True
        else:
            shape_ok = _same_type_neighbor_count(neighbors, terrain, index, type_index) == 1
    elif type_name in ("lake", "marsh", "desert"):
        # First two hexes of a round are free; from the third on, the
        # hex must be growing an existing body rather than sprouting a
        # new one.
        if placed_so_far < 2:
            shape_ok = True
        else:
            shape_ok = _same_type_neighbor_count(neighbors, terrain, index, type_index) >= 2
    else:
        shape_ok = True

    if not shape_ok:
        return False
    if check_disconnect and type_name in _IMPASSABLE_TYPES and _would_disconnect(grid, neighbors, terrain, index):
        return False
    return True


def _place_round(grid, neighbors, terrain, rng, start, type_name, round_index, log, bag, check_disconnect=True):
    """Returns [] without touching `bag` if `start` itself (the round's
    seed hex, which - unlike every later hex in the blob - isn't run
    through _can_place's shape rules either, by design) would disconnect
    the board; generate_terrain's loop just tries a fresh random type/
    start combo next iteration when that happens."""
    if check_disconnect and type_name in _IMPASSABLE_TYPES and _would_disconnect(grid, neighbors, terrain, start):
        return []

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
            for j in neighbors[h]:
                if j != -1 and terrain[j] == _UNSET:
                    candidates.add(j)
        candidates = [
            c for c in candidates
            if _can_place(grid, neighbors, terrain, c, type_name, type_index, len(placed), check_disconnect=check_disconnect)
        ]
        if not candidates:
            break
        choice = rng.choice(candidates)
        place(choice)
        placed.append(choice)
    bag[type_name] -= len(placed)
    return placed


def generate_terrain(grid, rng, log=None):
    """Builds a full terrain map (a plain Python list[int], one
    TERRAIN_TO_INDEX value per hex) in rounds: each round draws a hex
    type from a shrinking "bag" (weighted by how many of that type are
    left, see BAG_COUNTS) and grows a random-sized blob of it out from a
    spot touching the already-generated board (or, for the very first
    round, a random edge hex), subject to that type's placement rules -
    which include never letting a mountain/lake hex wall part of the
    board off into an unreachable island (see _would_disconnect):
    _can_place rejects any such hex mid-blob, and _place_round refuses to
    even seed a round on one, in which case this loop just tries a
    different random type/start combo next iteration - `start` gets
    redrawn from the same "adjacent to whatever's already placed"
    candidate pool every iteration regardless of whether anything
    actually got placed, so a blocked attempt costs nothing but a retry.

    Only genuinely pathological bag exhaustion (every passable-type entry
    spent while unset hexes remain, forcing mountain/lake attempts that
    keep landing somewhere disconnecting) could make that retry loop drag
    on; `max_stuck_rounds` is a generous, board-size-scaled circuit
    breaker that just stops enforcing the island check for the rest of
    generation if it's ever actually hit - an occasional island beats a
    hang, and this is orders of magnitude more attempts than the default
    BAG_COUNTS/RADIUS should ever need.

    If `log` is given (a list), every individual hex placement is
    appended to it in placement order as {"q","r","s","terrain","round"}
    - see hex_gen.py for a step-by-step visualizer built on that log."""
    neighbors = _neighbor_lists(grid)
    terrain = [_UNSET] * grid.num_hexes
    unset = set(range(grid.num_hexes))
    bag = dict(BAG_COUNTS)

    edge_hexes = [i for i in unset if grid.is_edge(i)]
    start = rng.choice(edge_hexes)

    round_index = 0
    stuck_rounds = 0
    max_stuck_rounds = grid.num_hexes * 4
    check_disconnect = True
    while unset:
        types = [t for t, count in bag.items() if count > 0]
        weights = [bag[t] for t in types]
        type_name = rng.choices(types, weights=weights, k=1)[0]
        placed = _place_round(grid, neighbors, terrain, rng, start, type_name, round_index, log, bag,
                               check_disconnect=check_disconnect)
        round_index += 1
        if placed:
            stuck_rounds = 0
            unset.difference_update(placed)
        else:
            stuck_rounds += 1
            if stuck_rounds >= max_stuck_rounds:
                check_disconnect = False
        if unset:
            candidates = [
                i for i in unset
                if any(terrain[j] != _UNSET for j in neighbors[i] if j != -1)
            ]
            start = rng.choice(candidates)

    return terrain


def create_initial_state(radius=8, num_factions=8, seed=42, terrain_log=None, device=None):
    """terrain_log: optional list - if given, receives every individual
    terrain-generation hex placement in order (see generate_terrain's
    docstring), for run.py to dump alongside board_state.json. Returns a
    batch_size=1 ArrayState - see module docstring for why this isn't
    batched, and state.py's stack_states for assembling B of these into a
    training batch."""
    rng = random.Random(seed)
    grid = HexGrid(radius, device=device)
    state = new_empty(grid, num_factions, batch_size=1, device=device)

    terrain = generate_terrain(grid, rng, log=terrain_log)
    state.terrain[0] = torch.tensor(terrain, dtype=state.terrain.dtype, device=state.device)

    state.gold[0, :] = STARTING_GOLD
    state.kill_xp[0, :] = STARTING_KILL_XP

    return state
