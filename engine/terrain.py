"""
Terrain effects applied once at the end of each full turn (after
movement, cavalry movement, and all battles have resolved).

  - Desert: any army that ends the full turn on a desert hex loses 1
    unit (infantry -> cavalry -> archers priority, same as battle deaths).
  - Marsh: armies frozen this turn (by entering a marsh hex) are
    unfrozen, ready to move again next turn.
"""

from .state import UNIT_TYPES, army_total


def apply_terrain_effects(state):
    for h in state.board.values():
        if h.army is None:
            continue

        if h.terrain == "desert":
            for ut in UNIT_TYPES:
                if h.army[ut] > 0:
                    h.army[ut] -= 1
                    break
            if army_total(h.army) == 0:
                h.army = None
                continue

        if h.army is not None and h.army.get("frozen"):
            h.army["frozen"] = False

    return state
