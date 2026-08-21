"""
Income phase for engine_v2 - ported from engine/income.py.

Mechanical only (no agent decisions): +3 silver/turn per faction, +1 per
city beyond the 2nd; a faction with zero cities gets no income and loses
one unit instead (see _remove_first_unit below).
"""

import numpy as np

from .state import NO_FACTION


def apply_income_phase(state):
    """Applies one turn's income in place. Iterates factions in id order
    (0..num_factions-1) - engine/income.py iterates state.players.items()
    in insertion order, which create_initial_state always builds as
    0..num_factions-1, so this matches."""
    for faction in range(state.num_factions):
        if not state.alive[faction]:
            continue
        cities = int(np.sum(state.city_owner == faction))
        if cities == 0:
            state.silver[faction] = 0
            _remove_first_unit(state, faction)
        else:
            bonus = max(0, cities - 2)
            state.silver[faction] += 3 + bonus

    return state


def _remove_first_unit(state, faction):
    """Removes one unit (infantry -> cavalry -> archers priority) from
    the FIRST hex (in board/hex-index order) holding a peaceful army for
    `faction` with any units - mirrors engine/income.py's
    _remove_one_unit_anywhere, including that it only looks at peaceful
    board armies, never units currently locked in a pending battle."""
    candidates = np.nonzero((state.army_faction == faction) & (state.army_units.sum(axis=1) > 0))[0]
    if len(candidates) == 0:
        return
    hex_index = int(candidates[0])

    for ut in range(3):  # infantry, cavalry, archers priority
        if state.army_units[hex_index, ut] > 0:
            state.army_units[hex_index, ut] -= 1
            break

    if int(state.army_units[hex_index].sum()) == 0:
        state.army_faction[hex_index] = NO_FACTION
        state.frozen[hex_index] = False
