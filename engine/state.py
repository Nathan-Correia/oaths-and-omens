"""
Torch-based, batched game state for engine.

Every per-hex/per-faction field carries a leading batch dimension B - an
ordinary single game (what run.py/the visualizer/tournament.py use) is
just B=1; training-time self-play uses B>1. There is no separate
"single-game mode" - B=1 is just the smallest batch. See engine/geometry.py
for why board TOPOLOGY (HexGrid.neighbor_table/coords_array) is never
batched: every game in one batch shares the same radius, so one topology
serves the whole batch. A batch is homogeneous - one radius, one
num_factions, shared by every game in it.

Board state (terrain/city/army/battle-contribution arrays) plus per-player
resources (gold, kill_xp, alive) and turn_number together make up the full
game state. voting_tokens isn't modeled: nothing in engine/ ever reads or
mutates it, so there's nothing to port.

Battle contributions are stored padded to a fixed MAX_BATTLE_CONTRIB slots
per hex - a variable-length list doesn't have a fixed shape a tensor op can
work on. Sized generously based on testing so far; battle resolution can
extend a battle's contributions further via cavalry dismounts, so this cap
may need revisiting if MAX_BATTLE_CONTRIB errors ever fire.

battle_order is the one piece of state that stays a plain Python
structure, not a tensor: list[list[int]], one inner list per batch item,
each holding that game's pending-battle hex indices in creation order.
When a turn resolves more than one battle IN THE SAME GAME, the per-faction
dismount infantry cap tally is *shared* across all of them (see
engine/turn.py's _run_battle_phase), so which battle gets processed first
can change the outcome in edge cases near the cap - order is real game
state, not cosmetic. Not vectorized/tensor-shaped, and deliberately so:
this is small, inherently sequential per-game bookkeeping, not a per-hex
board fact. movement.py's _start_or_extend_battle appends a hex to its
game's list the moment it newly locks; turn.py's battle phase (and
battle.py's rectify_overflow) remove a hex the moment it unlocks.
"""

import dataclasses
from dataclasses import dataclass

import torch

from .geometry import HexGrid

TERRAIN_TYPES = ["plains", "mountain", "lake", "desert", "marsh"]
TERRAIN_TO_INDEX = {t: i for i, t in enumerate(TERRAIN_TYPES)}
IMPASSABLE_TERRAIN_INDICES = torch.tensor(
    [TERRAIN_TO_INDEX["mountain"], TERRAIN_TO_INDEX["lake"]], dtype=torch.int8
)
# bool[len(TERRAIN_TYPES)]: IMPASSABLE_BY_TERRAIN[t] is True iff terrain
# type t is impassable - a fancy-index lookup (`IMPASSABLE_BY_TERRAIN
# [some_terrain_tensor]`) is a much cheaper way to test membership in this
# fixed 2-element set than an isin-style check, which pays real fixed
# overhead that dominates when called many times per turn against a tiny
# array - see engine/movement.py's _legal_mask, profiled at ~40% of a
# whole game's engine-level runtime before this (pre-rewrite finding,
# ported forward into this design rather than reintroducing the mistake).
IMPASSABLE_BY_TERRAIN = torch.zeros(len(TERRAIN_TYPES), dtype=torch.bool)
IMPASSABLE_BY_TERRAIN[IMPASSABLE_TERRAIN_INDICES.long()] = True

UNIT_TYPES = ["infantry", "cavalry", "archers"]  # index 0/1/2
MAX_STACK_SIZE = 6
MAX_BATTLE_CONTRIB = 16  # see module docstring
SPAWN_CAPS = torch.tensor([24, 12, 12], dtype=torch.int32)  # infantry, cavalry, archers

RESOURCE_TYPES = ["wood", "iron", "clay", "fish"]
RESOURCE_TO_INDEX = {t: i for i, t in enumerate(RESOURCE_TYPES)}

# An outpost's upgrade slot (at most one at a time - see engine/collect.py/
# engine/buy.py). NO_UPGRADE also covers every capital hex, and every hex
# with no city at all - outpost_upgrade is only ever meaningful where
# city_owner != NO_FACTION and is_capital is False.
UPGRADE_TYPES = ["barracks", "workshop", "temple"]
UPGRADE_TO_INDEX = {t: i for i, t in enumerate(UPGRADE_TYPES)}
NO_UPGRADE = -1

NO_FACTION = -1
NO_ORIGIN = -1

# Every per-hex/per-faction tensor field on ArrayState, in declaration
# order - used by stack_states/unstack_states/ArrayState.to so those never
# have to hardcode (and risk drifting from) the field list by hand.
_TENSOR_FIELDS = [
    "terrain", "city_owner", "is_capital", "outpost_upgrade", "city_placer",
    "capital_settle_order", "army_faction", "army_units", "frozen", "locked",
    "battle_faction", "battle_origin", "battle_units", "battle_moved", "battle_round",
    "gold", "resources", "kill_xp", "victory_points", "alive", "turn_number",
]


@dataclass
class ArrayState:
    grid: HexGrid                # unbatched - see module docstring
    terrain: torch.Tensor          # int8[B, N]          - index into TERRAIN_TYPES
    city_owner: torch.Tensor       # int8[B, N]           - NO_FACTION if no city; a capital or an outpost
    is_capital: torch.Tensor       # bool[B, N]           - only meaningful where city_owner != NO_FACTION
    outpost_upgrade: torch.Tensor  # int8[B, N]           - NO_UPGRADE, or an index into UPGRADE_TYPES;
                                    # only meaningful where city_owner != NO_FACTION and is_capital is False
    city_placer: torch.Tensor      # int8[B, N]           - which faction placed the colourless city here
                                    # during setup (see placement.py's run_city_setup); NO_FACTION if no
                                    # city was ever placed on this hex. Left populated (harmless) once
                                    # setup finishes - city_owner/is_capital are authoritative from then on.
    capital_settle_order: torch.Tensor  # int32[B, F]     - the order (0, 1, 2, ...) each faction's final
                                    # capital was settled in during the draft (see placement.py's
                                    # run_city_setup); -1 before setup runs. Used by turn.py's
                                    # get_game_winner to break a tied-VP finish.
    army_faction: torch.Tensor     # int8[B, N]           - NO_FACTION if no army
    army_units: torch.Tensor       # int16[B, N, 3]       - infantry, cavalry, archers
    frozen: torch.Tensor           # bool[B, N]
    locked: torch.Tensor           # bool[B, N]
    battle_faction: torch.Tensor   # int8[B, N, K]        - NO_FACTION for empty slots
    battle_origin: torch.Tensor    # int32[B, N, K]       - hex index units in this slot came from
    battle_units: torch.Tensor     # int16[B, N, K, 3]
    battle_moved: torch.Tensor     # bool[B, N, K]        - True iff this slot's units moved into this
                                    # hex to join the fight (attacker, encounter/line-battle participant,
                                    # or a later reinforcement); False only for the original stationary
                                    # occupant a battle triggered against. Gates the real Archers ability
                                    # (see battle.py's apply_archer_abilities) to the attacking side -
                                    # separate from battle_origin, which keeps meaning "where do this
                                    # slot's surviving/overflow units retreat to" and must NOT be
                                    # (mis)used to infer this, since a Line Battle's battle_hex tie-break
                                    # can coincidentally equal one side's own origin hex.
    battle_round: torch.Tensor     # int16[B, N]
    battle_order: list              # list[list[int]], length B - see module docstring
    gold: torch.Tensor             # int32[B, F]
    resources: torch.Tensor        # int32[B, F, 4]       - Wood/Iron/Clay/Fish, see RESOURCE_TO_INDEX
    kill_xp: torch.Tensor          # int32[B, F]
    victory_points: torch.Tensor   # int32[B, F]          - the win condition; see turn.py's VP_TO_WIN
    alive: torch.Tensor            # bool[B, F]           - vestigial now that elimination is gone (always
                                    # True, never set False); kept only so encode.py/hex_visualizer.py don't
                                    # need their own unrelated changes
    turn_number: torch.Tensor      # int32[B]             - per batch item: games in one batch can reach
                                    # game-end at different turn counts, so this isn't a single shared int
                                    # anymore (see engine/turn.py's check_game_end/get_game_winner).
    num_factions: int              # shared across the whole batch (homogeneous batch - see module docstring)

    @property
    def num_hexes(self):
        return self.grid.num_hexes

    @property
    def batch_size(self):
        return self.terrain.shape[0]

    @property
    def device(self):
        return self.terrain.device

    def to(self, device):
        """A new ArrayState with every tensor field (and the shared grid)
        moved to `device` - battle_order is plain Python, unaffected."""
        kwargs = {name: getattr(self, name).to(device) for name in _TENSOR_FIELDS}
        return dataclasses.replace(self, grid=self.grid.to(device), **kwargs)

    def clone(self):
        """A fully independent copy - every tensor field cloned, every
        battle_order inner list copied too (mutating the clone's
        battle_order must never touch the original's) - for simulating
        forward without touching the real game state. `grid` is shared by
        reference (board topology never mutates). Used by search-based
        agents (see agents/tactician_agent.py) to roll out candidate
        moves on a disposable copy."""
        kwargs = {name: getattr(self, name).clone() for name in _TENSOR_FIELDS}
        return dataclasses.replace(self, battle_order=[list(lst) for lst in self.battle_order], **kwargs)


def new_empty(grid, num_factions, batch_size=1, device=None):
    """A fresh all-empty ArrayState with the given batch size. `device`
    defaults to `grid`'s own device (typically CPU for per-game setup;
    pass a CUDA device explicitly for the batched turn-loop)."""
    device = torch.device(device) if device is not None else grid.device
    B, N = batch_size, grid.num_hexes
    return ArrayState(
        grid=grid,
        terrain=torch.zeros((B, N), dtype=torch.int8, device=device),
        city_owner=torch.full((B, N), NO_FACTION, dtype=torch.int8, device=device),
        is_capital=torch.zeros((B, N), dtype=torch.bool, device=device),
        outpost_upgrade=torch.full((B, N), NO_UPGRADE, dtype=torch.int8, device=device),
        city_placer=torch.full((B, N), NO_FACTION, dtype=torch.int8, device=device),
        capital_settle_order=torch.full((B, num_factions), -1, dtype=torch.int32, device=device),
        army_faction=torch.full((B, N), NO_FACTION, dtype=torch.int8, device=device),
        army_units=torch.zeros((B, N, 3), dtype=torch.int16, device=device),
        frozen=torch.zeros((B, N), dtype=torch.bool, device=device),
        locked=torch.zeros((B, N), dtype=torch.bool, device=device),
        battle_faction=torch.full((B, N, MAX_BATTLE_CONTRIB), NO_FACTION, dtype=torch.int8, device=device),
        battle_origin=torch.full((B, N, MAX_BATTLE_CONTRIB), NO_ORIGIN, dtype=torch.int32, device=device),
        battle_units=torch.zeros((B, N, MAX_BATTLE_CONTRIB, 3), dtype=torch.int16, device=device),
        battle_moved=torch.zeros((B, N, MAX_BATTLE_CONTRIB), dtype=torch.bool, device=device),
        battle_round=torch.zeros((B, N), dtype=torch.int16, device=device),
        battle_order=[[] for _ in range(B)],
        gold=torch.zeros((B, num_factions), dtype=torch.int32, device=device),
        resources=torch.zeros((B, num_factions, len(RESOURCE_TYPES)), dtype=torch.int32, device=device),
        kill_xp=torch.zeros((B, num_factions), dtype=torch.int32, device=device),
        victory_points=torch.zeros((B, num_factions), dtype=torch.int32, device=device),
        alive=torch.ones((B, num_factions), dtype=torch.bool, device=device),
        turn_number=torch.zeros(B, dtype=torch.int32, device=device),
        num_factions=num_factions,
    )


def stack_states(states):
    """[ArrayState, ...] (any batch sizes, same grid/num_factions/device)
    -> one ArrayState with batch dim = the sum of the inputs' batch sizes.
    Used to assemble a training batch out of independently-generated
    single games (engine/setup.py's terrain gen + engine/placement.py's
    draft stay per-game CPU work - see those modules - producing B=1
    states that get stacked here after setup finishes)."""
    if not states:
        raise ValueError("stack_states requires at least one state")
    grid = states[0].grid
    num_factions = states[0].num_factions
    for s in states[1:]:
        if s.grid.num_hexes != grid.num_hexes or s.num_factions != num_factions:
            raise ValueError("stack_states requires every input to share the same board size and num_factions")

    kwargs = {name: torch.cat([getattr(s, name) for s in states], dim=0) for name in _TENSOR_FIELDS}
    battle_order = [lst for s in states for lst in s.battle_order]
    return ArrayState(grid=grid, num_factions=num_factions, battle_order=battle_order, **kwargs)


def unstack_states(state):
    """Inverse of stack_states: one ArrayState with batch dim B -> a list
    of B ArrayStates, each with batch dim 1. Used wherever a single game
    needs to be pulled out of a batch for per-game tooling (logging,
    visualization, an agent that only ever sees one game at a time)."""
    out = []
    for b in range(state.batch_size):
        kwargs = {name: getattr(state, name)[b : b + 1] for name in _TENSOR_FIELDS}
        out.append(ArrayState(
            grid=state.grid, num_factions=state.num_factions,
            battle_order=[list(state.battle_order[b])], **kwargs,
        ))
    return out


def count_units_in_play(state, faction, unit_index):
    """int32[B] - how many of `faction`'s unit_index-typed units
    (0=infantry, 1=cavalry, 2=archers) currently exist, on the board or
    mid-battle, for every batch item at once. `faction` is a plain int
    applied uniformly across the batch (num_factions is batch-wide, so a
    faction index means the same thing in every game of one batch).
    Prefer a caller-maintained running tally over calling this in a loop
    (the same "don't rescan every check" lesson as before the rewrite);
    this is fine for one-off checks."""
    board_mask = state.army_faction == faction  # [B, N]
    board_total = (state.army_units[..., unit_index] * board_mask).sum(dim=1)
    battle_mask = state.battle_faction == faction  # [B, N, K]
    battle_total = (state.battle_units[..., unit_index] * battle_mask).sum(dim=(1, 2))
    return (board_total + battle_total).to(torch.int32)
