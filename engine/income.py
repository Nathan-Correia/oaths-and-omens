"""
Income phase: no agent decisions involved, purely mechanical.

Rules encoded here:
  - Base income: 3 silver/turn.
  - +1 silver per city beyond the 2nd (i.e. 3rd city and up).
  - A player with 0 cities gets 0 income and loses 1 unit instead
    (removed in a fixed infantry -> cavalry -> archers priority, same
    as battle-death priority, for consistency).
"""

from .state import UNIT_TYPES, army_total


def _count_cities(state, faction):
    return sum(1 for h in state.board.values() if h.city_owner == faction)


def _remove_one_unit_anywhere(state, faction):
    """Used for the 'no cities -> lose 1 unit/turn' penalty. Picks the
    first army belonging to `faction` found on the board and removes
    one unit from it, infantry -> cavalry -> archers priority."""
    for hex_state in state.board.values():
        army = hex_state.army
        if army and army["faction"] == faction and army_total(army) > 0:
            for ut in UNIT_TYPES:
                if army[ut] > 0:
                    army[ut] -= 1
                    if army_total(army) == 0:
                        hex_state.army = None
                    return
    # no units left anywhere - nothing to remove


def apply_income_phase(state):
    for faction, player in state.players.items():
        if not player.alive:
            continue
        cities = _count_cities(state, faction)
        if cities == 0:
            player.silver = 0
            _remove_one_unit_anywhere(state, faction)
        else:
            bonus = max(0, cities - 2)
            player.silver += 3 + bonus
    return state
