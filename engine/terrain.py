"""
Batched terrain effects for engine - applied once at the end of each
game's turn (after movement, cavalry movement, and battles have all
resolved), across every game in the batch at once:

  - Desert: any army ending the turn on a desert hex with no city on it
    loses 1 unit (infantry -> cavalry -> archers priority).
  - Marsh: armies frozen this turn (by entering a marsh hex) unfreeze,
    ready to move again next turn.

Fully vectorized - no Python loop over hexes or batch items. "First
nonzero unit type" (the death-priority cascade) is an argmax over a
boolean has-units mask instead of a per-hex Python loop.
"""

import torch

from .state import NO_FACTION, TERRAIN_TO_INDEX

DESERT_INDEX = TERRAIN_TO_INDEX["desert"]


def apply_terrain_effects(state):
    has_army = state.army_faction != NO_FACTION  # [B, N]
    needs_desert_loss = has_army & (state.terrain == DESERT_INDEX) & (state.city_owner == NO_FACTION)

    has_type = state.army_units > 0  # [B, N, 3]
    first_type = has_type.long().argmax(dim=-1)  # [B, N] - first (infantry=0 first) nonzero type

    b_idx, h_idx = torch.nonzero(needs_desert_loss, as_tuple=True)
    if len(b_idx):
        state.army_units[b_idx, h_idx, first_type[b_idx, h_idx]] -= 1

    now_empty = needs_desert_loss & (state.army_units.sum(dim=-1) == 0)
    state.army_faction[now_empty] = NO_FACTION
    state.frozen[now_empty] = False

    # Unfreeze whatever's left standing. Recomputing has_army (rather than
    # reusing needs_desert_loss/has_army from above) matters: a hex the
    # desert loss just emptied must NOT be touched here.
    still_has_army = state.army_faction != NO_FACTION
    to_unfreeze = still_has_army & state.frozen
    state.frozen[to_unfreeze] = False

    return state
