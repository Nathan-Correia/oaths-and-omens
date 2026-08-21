"""
Array-based game state for engine_v2.

engine/state.py represents the board as {(q, r, s): HexState} - a dict of
dataclass objects, each holding a terrain string, an optional army dict,
etc. engine_v2 instead holds every hex's data as a slice of a handful of
fixed-shape numpy arrays, indexed by the HexGrid from geometry.py. This is
what a masked/vectorized engine actually needs: dicts and variable-length
per-hex objects don't have a "shape" a tensor op can work on, but
`army_units[hex_index]` does.

Board state (terrain/city/army/battle-contribution arrays) plus per-player
resources (silver, kill_xp, alive) and turn_number together make up the
full game state - the same split as v1's GameState (board dict) +
PlayerState dict + turn_number, just array-shaped. voting_tokens isn't
modeled: nothing in engine/ ever reads or mutates it (voting is stubbed
for v1 per turn.py's tally_final_score), so there's nothing to port.

Battle contributions are stored padded to a fixed MAX_BATTLE_CONTRIB slots
per hex (engine/state.py's Battle.contributions is a variable-length
list - same fixed-shape-over-variable-length trade as everywhere else in
engine_v2). Sized generously based on testing so far; battle resolution
can extend a battle's contributions further via cavalry dismounts, so
this cap may need revisiting if MAX_BATTLE_CONTRIB errors ever fire.

battle_order is the one piece of state that's a plain Python list, not an
array: when a turn resolves more than one battle, the per-faction
dismount infantry cap tally is *shared* across all of them (see
engine/turn.py's _run_battle_phase), so which battle gets processed first
can change the outcome in edge cases near the cap. v1 gets this for free
from dict insertion order (state.battles); engine_v2 has to track it
explicitly. movement.py's _start_or_extend_battle appends a hex the
moment it newly locks; turn.py's battle phase (and battle.py's
rectify_overflow) remove a hex the moment it unlocks. Not vectorized/
tensor-shaped, and deliberately so for now - this is a small, inherently
sequential piece of bookkeeping, not a per-hex board fact.
"""

from dataclasses import dataclass, field

import numpy as np

from .geometry import HexGrid

TERRAIN_TYPES = ["plains", "forest", "mountain", "lake", "desert", "marsh"]
TERRAIN_TO_INDEX = {t: i for i, t in enumerate(TERRAIN_TYPES)}
IMPASSABLE_TERRAIN_INDICES = np.array(
    [TERRAIN_TO_INDEX["mountain"], TERRAIN_TO_INDEX["lake"]], dtype=np.int8
)

UNIT_TYPES = ["infantry", "cavalry", "archers"]  # index 0/1/2, matches engine/state.py's order
MAX_STACK_SIZE = 6
MAX_BATTLE_CONTRIB = 16  # see module docstring
SPAWN_CAPS = np.array([24, 12, 12], dtype=np.int32)  # infantry, cavalry, archers - matches engine/state.py

NO_FACTION = -1
NO_ORIGIN = -1


@dataclass
class ArrayState:
    grid: HexGrid
    terrain: np.ndarray          # int8[N]            - index into TERRAIN_TYPES
    city_owner: np.ndarray       # int8[N]             - NO_FACTION if no city
    army_faction: np.ndarray     # int8[N]             - NO_FACTION if no army
    army_units: np.ndarray       # int16[N, 3]         - infantry, cavalry, archers
    frozen: np.ndarray           # bool_[N]
    locked: np.ndarray           # bool_[N]
    battle_faction: np.ndarray   # int8[N, K]          - NO_FACTION for empty slots
    battle_origin: np.ndarray    # int32[N, K]         - hex index units in this slot came from
    battle_units: np.ndarray     # int16[N, K, 3]
    battle_round: np.ndarray     # int16[N]
    battle_order: list           # [hex_index, ...] in battle-creation order - see module docstring
    silver: np.ndarray           # int32[num_factions]
    kill_xp: np.ndarray          # int32[num_factions]
    alive: np.ndarray            # bool_[num_factions]
    turn_number: int
    num_factions: int

    @property
    def num_hexes(self):
        return self.grid.num_hexes


def new_empty(grid, num_factions):
    n = grid.num_hexes
    return ArrayState(
        grid=grid,
        terrain=np.zeros(n, dtype=np.int8),
        city_owner=np.full(n, NO_FACTION, dtype=np.int8),
        army_faction=np.full(n, NO_FACTION, dtype=np.int8),
        army_units=np.zeros((n, 3), dtype=np.int16),
        frozen=np.zeros(n, dtype=bool),
        locked=np.zeros(n, dtype=bool),
        battle_faction=np.full((n, MAX_BATTLE_CONTRIB), NO_FACTION, dtype=np.int8),
        battle_origin=np.full((n, MAX_BATTLE_CONTRIB), NO_ORIGIN, dtype=np.int32),
        battle_units=np.zeros((n, MAX_BATTLE_CONTRIB, 3), dtype=np.int16),
        battle_round=np.zeros(n, dtype=np.int16),
        battle_order=[],
        silver=np.zeros(num_factions, dtype=np.int32),
        kill_xp=np.zeros(num_factions, dtype=np.int32),
        alive=np.ones(num_factions, dtype=bool),
        turn_number=0,
        num_factions=num_factions,
    )


def count_units_in_play(state, faction, unit_index):
    """How many of `faction`'s unit_index-typed units (0=infantry,
    1=cavalry, 2=archers) currently exist, on the board or mid-battle -
    mirrors engine/state.py's count_units_in_play. Prefer a caller-
    maintained running tally over calling this in a loop (see
    engine/state.py's note - the same "don't rescan every check" lesson
    applies here); this is fine for one-off checks."""
    board_total = int(state.army_units[state.army_faction == faction, unit_index].sum())
    battle_total = int(state.battle_units[state.battle_faction == faction, unit_index].sum())
    return board_total + battle_total


def from_v1(v1_state, num_factions, grid=None):
    """Builds an ArrayState from a v1 engine.state.GameState - board,
    battles, players, and turn_number. `grid` can be passed in to reuse
    an existing HexGrid (they're immutable per radius); otherwise one is
    built from v1_state.radius."""
    grid = grid or HexGrid(v1_state.radius)
    s = new_empty(grid, num_factions)

    for coord, h in v1_state.board.items():
        i = grid.index_of(coord)
        s.terrain[i] = TERRAIN_TO_INDEX[h.terrain]
        s.city_owner[i] = h.city_owner if h.city_owner is not None else NO_FACTION
        s.locked[i] = h.locked
        if h.army is not None:
            s.army_faction[i] = h.army["faction"]
            s.army_units[i] = [h.army[ut] for ut in UNIT_TYPES]
            s.frozen[i] = bool(h.army.get("frozen", False))

    for coord, battle in v1_state.battles.items():
        i = grid.index_of(coord)
        if len(battle.contributions) > MAX_BATTLE_CONTRIB:
            raise ValueError(
                f"battle at {coord} has {len(battle.contributions)} contributions, "
                f"exceeds MAX_BATTLE_CONTRIB={MAX_BATTLE_CONTRIB}"
            )
        for k, c in enumerate(battle.contributions):
            s.battle_faction[i, k] = c["faction"]
            s.battle_origin[i, k] = grid.index_of(c["origin_hex"])
            s.battle_units[i, k] = [c[ut] for ut in UNIT_TYPES]
        s.battle_round[i] = battle.round_number
        s.battle_order.append(i)  # preserves v1_state.battles' own insertion order

    for faction, player in v1_state.players.items():
        s.silver[faction] = player.silver
        s.kill_xp[faction] = player.kill_xp
        s.alive[faction] = player.alive
    s.turn_number = v1_state.turn_number

    return s


# Backwards-compatible alias for the board-only converter name used while
# only movement.py existed.
from_v1_board = from_v1


def state_diff(v1_state, v2_state):
    """Compares a v1 GameState against a v2 ArrayState - board, battles,
    players, turn_number. Returns a list of human-readable mismatch
    descriptions (empty list = equal). Used by the parity tests instead
    of a blanket equality check, so a failure says *what* differs
    instead of just that something does.
    """
    grid = v2_state.grid
    mismatches = []

    for faction, player in v1_state.players.items():
        if int(v2_state.silver[faction]) != player.silver:
            mismatches.append(f"faction {faction}: silver v1={player.silver} v2={v2_state.silver[faction]}")
        if int(v2_state.kill_xp[faction]) != player.kill_xp:
            mismatches.append(f"faction {faction}: kill_xp v1={player.kill_xp} v2={v2_state.kill_xp[faction]}")
        if bool(v2_state.alive[faction]) != player.alive:
            mismatches.append(f"faction {faction}: alive v1={player.alive} v2={v2_state.alive[faction]}")
    if v2_state.turn_number != v1_state.turn_number:
        mismatches.append(f"turn_number v1={v1_state.turn_number} v2={v2_state.turn_number}")

    for coord, h in v1_state.board.items():
        i = grid.index_of(coord)

        v1_terrain = TERRAIN_TO_INDEX[h.terrain]
        if v2_state.terrain[i] != v1_terrain:
            mismatches.append(f"{coord}: terrain v1={h.terrain} v2={TERRAIN_TYPES[v2_state.terrain[i]]}")

        v1_city = h.city_owner if h.city_owner is not None else NO_FACTION
        if v2_state.city_owner[i] != v1_city:
            mismatches.append(f"{coord}: city_owner v1={v1_city} v2={v2_state.city_owner[i]}")

        if bool(h.locked) != bool(v2_state.locked[i]):
            mismatches.append(f"{coord}: locked v1={h.locked} v2={v2_state.locked[i]}")

        if h.army is None:
            if v2_state.army_faction[i] != NO_FACTION:
                mismatches.append(f"{coord}: v1 has no army, v2 has faction {v2_state.army_faction[i]}")
        else:
            if v2_state.army_faction[i] != h.army["faction"]:
                mismatches.append(f"{coord}: army_faction v1={h.army['faction']} v2={v2_state.army_faction[i]}")
            v1_units = [h.army[ut] for ut in UNIT_TYPES]
            if list(v2_state.army_units[i]) != v1_units:
                mismatches.append(f"{coord}: army_units v1={v1_units} v2={list(v2_state.army_units[i])}")
            if bool(h.army.get("frozen", False)) != bool(v2_state.frozen[i]):
                mismatches.append(f"{coord}: frozen v1={h.army.get('frozen', False)} v2={v2_state.frozen[i]}")

    v1_battle_coords = set(v1_state.battles.keys())
    v2_battle_coords = {grid.coord_of(i) for i in np.nonzero(v2_state.locked)[0]}
    if v1_battle_coords != v2_battle_coords:
        mismatches.append(f"battle hex sets differ: v1={v1_battle_coords} v2={v2_battle_coords}")

    if len(v2_state.battle_order) != len(set(v2_state.battle_order)):
        mismatches.append(f"battle_order has duplicates: {v2_state.battle_order}")
    v2_battle_order_coords = {grid.coord_of(i) for i in v2_state.battle_order}
    if v2_battle_order_coords != v2_battle_coords:
        mismatches.append(
            f"battle_order doesn't match locked hexes: order={v2_battle_order_coords} locked={v2_battle_coords}"
        )

    for coord in v1_battle_coords & v2_battle_coords:
        i = grid.index_of(coord)
        battle = v1_state.battles[coord]

        v1_contribs = sorted(
            (c["faction"], grid.index_of(c["origin_hex"]), c["infantry"], c["cavalry"], c["archers"])
            for c in battle.contributions
        )
        used = np.nonzero(v2_state.battle_faction[i] != NO_FACTION)[0]
        v2_contribs = sorted(
            (
                int(v2_state.battle_faction[i, k]),
                int(v2_state.battle_origin[i, k]),
                int(v2_state.battle_units[i, k, 0]),
                int(v2_state.battle_units[i, k, 1]),
                int(v2_state.battle_units[i, k, 2]),
            )
            for k in used
        )
        if v1_contribs != v2_contribs:
            mismatches.append(f"{coord}: battle contributions differ v1={v1_contribs} v2={v2_contribs}")

    return mismatches


# Backwards-compatible alias for the board-only diff name used while only
# movement.py existed.
board_diff = state_diff
