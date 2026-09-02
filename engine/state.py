"""
Array-based game state for engine.

engine/state.py represents the board as {(q, r, s): HexState} - a dict of
dataclass objects, each holding a terrain string, an optional army dict,
etc. engine instead holds every hex's data as a slice of a handful of
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
engine). Sized generously based on testing so far; battle resolution
can extend a battle's contributions further via cavalry dismounts, so
this cap may need revisiting if MAX_BATTLE_CONTRIB errors ever fire.

battle_order is the one piece of state that's a plain Python list, not an
array: when a turn resolves more than one battle, the per-faction
dismount infantry cap tally is *shared* across all of them (see
engine/turn.py's _run_battle_phase), so which battle gets processed first
can change the outcome in edge cases near the cap. v1 gets this for free
from dict insertion order (state.battles); engine has to track it
explicitly. movement.py's _start_or_extend_battle appends a hex the
moment it newly locks; turn.py's battle phase (and battle.py's
rectify_overflow) remove a hex the moment it unlocks. Not vectorized/
tensor-shaped, and deliberately so for now - this is a small, inherently
sequential piece of bookkeeping, not a per-hex board fact.
"""

from dataclasses import dataclass, field

import numpy as np

from .geometry import HexGrid

TERRAIN_TYPES = ["plains", "mountain", "lake", "desert", "marsh"]
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
    city_owner: np.ndarray       # int8[N]             - NO_FACTION if no city; a capital or an outpost
    is_capital: np.ndarray       # bool_[N]            - only meaningful where city_owner != NO_FACTION
    city_placer: np.ndarray      # int8[N]             - which faction placed the colourless city here
                                  # during setup (see placement.py's run_city_setup); NO_FACTION if no
                                  # city was ever placed on this hex. Left populated (harmless) once
                                  # setup finishes - city_owner/is_capital are authoritative from then on.
    capital_settle_order: np.ndarray  # int32[num_factions] - the order (0, 1, 2, ...) each faction's
                                  # final capital was settled in during the draft (see
                                  # placement.py's run_city_setup); -1 before setup runs. Used by
                                  # turn.py's get_game_winner to break a tied-VP finish.
    army_faction: np.ndarray     # int8[N]             - NO_FACTION if no army
    army_units: np.ndarray       # int16[N, 3]         - infantry, cavalry, archers
    frozen: np.ndarray           # bool_[N]
    locked: np.ndarray           # bool_[N]
    battle_faction: np.ndarray   # int8[N, K]          - NO_FACTION for empty slots
    battle_origin: np.ndarray    # int32[N, K]         - hex index units in this slot came from
    battle_units: np.ndarray     # int16[N, K, 3]
    battle_moved: np.ndarray     # bool_[N, K]         - True iff this slot's units moved into this
                                  # hex to join the fight (attacker, encounter/line-battle participant,
                                  # or a later reinforcement); False only for the original stationary
                                  # occupant a battle triggered against. Gates the real Archers ability
                                  # (see battle.py's apply_archer_abilities) to the attacking side -
                                  # separate from battle_origin, which keeps meaning "where do this
                                  # slot's surviving/overflow units retreat to" and must NOT be
                                  # (mis)used to infer this, since a Line Battle's battle_hex tie-break
                                  # can coincidentally equal one side's own origin hex.
    battle_round: np.ndarray     # int16[N]
    battle_order: list           # [hex_index, ...] in battle-creation order - see module docstring
    silver: np.ndarray           # int32[num_factions]
    kill_xp: np.ndarray          # int32[num_factions]
    victory_points: np.ndarray   # int32[num_factions] - the win condition; see turn.py's VP_TO_WIN
    alive: np.ndarray            # bool_[num_factions] - vestigial now that elimination is gone (always
                                  # True, never set False); kept only so encode.py/hex_visualizer.py don't
                                  # need their own unrelated changes
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
        is_capital=np.zeros(n, dtype=bool),
        city_placer=np.full(n, NO_FACTION, dtype=np.int8),
        capital_settle_order=np.full(num_factions, -1, dtype=np.int32),
        army_faction=np.full(n, NO_FACTION, dtype=np.int8),
        army_units=np.zeros((n, 3), dtype=np.int16),
        frozen=np.zeros(n, dtype=bool),
        locked=np.zeros(n, dtype=bool),
        battle_faction=np.full((n, MAX_BATTLE_CONTRIB), NO_FACTION, dtype=np.int8),
        battle_origin=np.full((n, MAX_BATTLE_CONTRIB), NO_ORIGIN, dtype=np.int32),
        battle_units=np.zeros((n, MAX_BATTLE_CONTRIB, 3), dtype=np.int16),
        battle_moved=np.zeros((n, MAX_BATTLE_CONTRIB), dtype=bool),
        battle_round=np.zeros(n, dtype=np.int16),
        battle_order=[],
        silver=np.zeros(num_factions, dtype=np.int32),
        kill_xp=np.zeros(num_factions, dtype=np.int32),
        victory_points=np.zeros(num_factions, dtype=np.int32),
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

