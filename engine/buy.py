"""
Buy phase.

Two kinds of purchases:
  - buy_infantry: spend 2 silver at an owned city with no adjacent
    enemy army, subject to the 24-infantry concurrent cap (how many
    infantry this faction currently has in play at once - not a
    lifetime production count; losses free up capacity immediately).
  - convert_to_special: spend 1 kill-XP (a plain currency count, not
    literal tokens) + 1 silver to convert an existing infantry unit
    (anywhere you control one) into a cavalry or archer unit, subject
    to the same kind of concurrent 12/12 caps. This is the only way
    to acquire cavalry/archers - they can't be bought with silver.

Legal-action generation returns *atomic* single-unit actions (buy one
infantry, convert one unit); an agent can submit a list of several in
one buy phase, applied in order against the same faction's own
resources.
"""

from .geometry import hex_neighbors
from .state import SPAWN_CAPS, army_total, count_units_in_play

INFANTRY_COST = 2


def _adjacent_enemy_present(state, city_hex, faction):
    for n in hex_neighbors(city_hex, state.radius):
        h = state.board.get(n)
        if h and h.army and h.army["faction"] != faction:
            return True
    return False


def _remaining_cap(state, faction, unit_type):
    """Concurrent cap: SPAWN_CAPS[unit_type] minus how many of that type
    this faction currently has in play (on the board or mid-battle) -
    a unit dying frees up a slot immediately, unlike a lifetime count."""
    return SPAWN_CAPS[unit_type] - count_units_in_play(state, faction, unit_type)


def get_legal_buy_actions(state, faction):
    player = state.players[faction]
    actions = []

    if _remaining_cap(state, faction, "infantry") > 0 and player.silver >= INFANTRY_COST:
        for coord, h in state.board.items():
            if h.city_owner == faction and not h.locked:
                if not _adjacent_enemy_present(state, coord, faction):
                    actions.append({"type": "buy_infantry", "city_hex": coord})

    if player.kill_xp > 0 and player.silver >= 1:
        for coord, h in state.board.items():
            if h.army and h.army["faction"] == faction and h.army["infantry"] > 0:
                for unit_type in ("cavalry", "archers"):
                    if _remaining_cap(state, faction, unit_type) > 0:
                        actions.append({"type": "convert_to_special", "hex": coord, "unit_type": unit_type})

    return actions


def _apply_one(state, faction, action):
    player = state.players[faction]

    if action["type"] == "buy_infantry":
        coord = tuple(action["city_hex"])
        h = state.board.get(coord)
        if not h or h.city_owner != faction or h.locked:
            return False
        if _adjacent_enemy_present(state, coord, faction):
            return False
        if player.silver < INFANTRY_COST or _remaining_cap(state, faction, "infantry") <= 0:
            return False
        player.silver -= INFANTRY_COST
        if h.army is None:
            h.army = {"faction": faction, "infantry": 0, "cavalry": 0, "archers": 0, "frozen": False}
        if h.army["faction"] != faction or army_total(h.army) >= 6:
            return False  # can't spawn into an enemy tile or an already-full stack
        h.army["infantry"] += 1
        return True

    elif action["type"] == "convert_to_special":
        coord = tuple(action["hex"])
        unit_type = action["unit_type"]
        h = state.board.get(coord)
        if not h or not h.army or h.army["faction"] != faction or h.army["infantry"] <= 0:
            return False
        if player.kill_xp <= 0 or player.silver < 1 or _remaining_cap(state, faction, unit_type) <= 0:
            return False
        player.kill_xp -= 1
        player.silver -= 1
        h.army["infantry"] -= 1
        h.army[unit_type] += 1
        return True

    return False


def apply_buy_phase(state, actions_by_faction):
    """actions_by_faction: {faction_id: [action, ...]}. Applied faction
    by faction, in order within each faction's own list; factions only
    ever touch their own resources during this phase so ordering across
    factions doesn't matter."""
    for faction, actions in actions_by_faction.items():
        for action in actions:
            _apply_one(state, faction, action)
    return state