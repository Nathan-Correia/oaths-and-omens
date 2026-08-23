"""
Terrain effects for engine - ported from engine/terrain.py. Applied
once at the end of each full turn (after movement, cavalry movement, and
battles have all resolved):

  - Desert: any army ending the turn on a desert hex with no city on it
    loses 1 unit (infantry -> cavalry -> archers priority).
  - Marsh: armies frozen this turn (by entering a marsh hex) unfreeze,
    ready to move again next turn.
"""

import numpy as np

from .state import NO_FACTION, TERRAIN_TO_INDEX

DESERT_INDEX = TERRAIN_TO_INDEX["desert"]


def apply_terrain_effects(state):
    has_army = state.army_faction != NO_FACTION
    needs_desert_loss = has_army & (state.terrain == DESERT_INDEX) & (state.city_owner == NO_FACTION)

    for hex_index in np.nonzero(needs_desert_loss)[0]:
        hex_index = int(hex_index)
        for ut in range(3):  # infantry, cavalry, archers priority
            if state.army_units[hex_index, ut] > 0:
                state.army_units[hex_index, ut] -= 1
                break
        if int(state.army_units[hex_index].sum()) == 0:
            state.army_faction[hex_index] = NO_FACTION
            state.frozen[hex_index] = False

    # Unfreeze whatever's left standing. Recomputing has_army (rather than
    # reusing the value from above) matters: a hex the desert loss just
    # emptied must NOT be touched here, mirroring engine/terrain.py's
    # `continue` that skips the unfreeze check for a hex wiped out this
    # same pass.
    still_has_army = state.army_faction != NO_FACTION
    to_unfreeze = still_has_army & state.frozen
    state.frozen[to_unfreeze] = False

    return state
