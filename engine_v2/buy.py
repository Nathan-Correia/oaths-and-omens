"""
Buy phase for engine_v2 - ported from engine/buy.py.

Two kinds of purchases, atomic (one unit each) same as v1: buy_infantry
(spend 2 silver at an owned, undefended-by-adjacent-enemy city) and
convert_to_special (spend 1 kill-XP + 1 silver to convert an existing
infantry unit into cavalry/archers), both subject to the 24/12/12
concurrent SPAWN_CAPS.

SCOPE: actions use the same atomic representation as v1 (hex identified
by index instead of coordinate) rather than the fixed/masked action-space
design discussed for the eventual RL policy - that design is tied to the
actual agent interface, which is a separate milestone from getting these
mechanics correct. Unit types are referenced by index here (0=infantry,
1=cavalry, 2=archers) matching state.UNIT_TYPES/SPAWN_CAPS order.
"""

import numpy as np

from .state import NO_FACTION, SPAWN_CAPS

INFANTRY_COST = 2
CAVALRY = 1
ARCHERS = 2


def count_all_units_in_play(state, faction):
    """[3] array: how many of `faction`'s infantry/cavalry/archers
    currently exist, on the board or mid-battle - mirrors
    engine/state.py's count_all_units_in_play."""
    board_total = state.army_units[state.army_faction == faction].sum(axis=0)
    battle_total = state.battle_units[state.battle_faction == faction].sum(axis=0)
    return board_total + battle_total


def _remaining_cap(counts, unit_index):
    return int(SPAWN_CAPS[unit_index] - counts[unit_index])


def _adjacent_enemy_present(state, hex_index, faction):
    neighbors = state.grid.neighbor_table[hex_index]
    valid = neighbors >= 0
    safe_neighbors = np.where(valid, neighbors, 0)
    neighbor_faction = state.army_faction[safe_neighbors]
    enemy = valid & (neighbor_faction != NO_FACTION) & (neighbor_faction != faction)
    return bool(enemy.any())


def get_legal_buy_actions(state, faction):
    counts = count_all_units_in_play(state, faction)
    actions = []

    if _remaining_cap(counts, 0) > 0 and state.silver[faction] >= INFANTRY_COST:
        city_hexes = np.nonzero((state.city_owner == faction) & ~state.locked)[0]
        for hex_index in city_hexes:
            hex_index = int(hex_index)
            if not _adjacent_enemy_present(state, hex_index, faction):
                actions.append({"type": "buy_infantry", "city_hex": hex_index})

    if state.kill_xp[faction] > 0 and state.silver[faction] >= 1:
        army_hexes = np.nonzero((state.army_faction == faction) & (state.army_units[:, 0] > 0))[0]
        for hex_index in army_hexes:
            hex_index = int(hex_index)
            for unit_index, unit_name in ((CAVALRY, "cavalry"), (ARCHERS, "archers")):
                if _remaining_cap(counts, unit_index) > 0:
                    actions.append({"type": "convert_to_special", "hex": hex_index, "unit_type": unit_name})

    return actions


def _apply_one(state, faction, action, counts, enemy_adjacent_cache):
    if action["type"] == "buy_infantry":
        hex_index = action["city_hex"]
        if state.city_owner[hex_index] != faction or state.locked[hex_index]:
            return False

        adjacent_enemy = enemy_adjacent_cache.get(hex_index)
        if adjacent_enemy is None:
            adjacent_enemy = _adjacent_enemy_present(state, hex_index, faction)
            enemy_adjacent_cache[hex_index] = adjacent_enemy
        if adjacent_enemy:
            return False

        if state.silver[faction] < INFANTRY_COST or _remaining_cap(counts, 0) <= 0:
            return False
        state.silver[faction] -= INFANTRY_COST

        if state.army_faction[hex_index] == NO_FACTION:
            state.army_faction[hex_index] = faction
        # Ported as-is from engine/buy.py: can't actually trigger given the
        # city_owner==faction check above (a hostile army parked on your
        # own city would already have flipped city_owner via
        # _maybe_capture_city), but kept for exact behavioral parity.
        if state.army_faction[hex_index] != faction or int(state.army_units[hex_index].sum()) >= 6:
            return False

        state.army_units[hex_index, 0] += 1
        counts[0] += 1
        return True

    elif action["type"] == "convert_to_special":
        hex_index = action["hex"]
        unit_index = CAVALRY if action["unit_type"] == "cavalry" else ARCHERS
        if state.army_faction[hex_index] != faction or state.army_units[hex_index, 0] <= 0:
            return False
        if state.kill_xp[faction] <= 0 or state.silver[faction] < 1 or _remaining_cap(counts, unit_index) <= 0:
            return False

        state.kill_xp[faction] -= 1
        state.silver[faction] -= 1
        state.army_units[hex_index, 0] -= 1
        state.army_units[hex_index, unit_index] += 1
        counts[0] -= 1
        counts[unit_index] += 1
        return True

    return False


def apply_buy_phase(state, actions_by_faction):
    """actions_by_faction: {faction: [action, ...]}. Same caching pattern
    as engine/buy.py: one count snapshot + one enemy-adjacency cache per
    faction, maintained locally rather than rescanned per action."""
    for faction, actions in actions_by_faction.items():
        counts = count_all_units_in_play(state, faction)
        enemy_adjacent_cache = {}
        for action in actions:
            _apply_one(state, faction, action, counts, enemy_adjacent_cache)
    return state
