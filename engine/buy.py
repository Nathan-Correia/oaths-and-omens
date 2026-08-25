"""
Buy phase for engine - ported from engine/buy.py, plus a new third
purchase kind (build_outpost) for the outposts/VP ruleset.

Three kinds of purchases, atomic (one unit each): buy_infantry (spend 2
silver at an owned city - capital or outpost, both use city_owner the
same way), convert_to_special (spend 1 kill-XP + 1 silver to convert an
existing infantry unit into cavalry/archers), and build_outpost (spend 3
silver + consume 1 unit already standing on the target hex to found a
new outpost there). The first two are subject to the 24/12/12 concurrent
SPAWN_CAPS; build_outpost is subject to OUTPOST_CAP instead (how many
outposts one faction may have standing at once) plus the
placement-distance rules in _can_build_outpost.

RULE CHANGE - siege: buy_infantry's undefended-by-adjacent-enemy
requirement only applies at an outpost now, not a capital - a capital
can always recruit infantry regardless of what's adjacent to it (see
_adjacent_enemy_present's call sites below, both now gated on
`not is_capital`).

Recruiting at an outpost (buy_infantry there) is capped at 1 per turn -
enforced in apply_buy_phase, not here, since it's a per-turn-batch
property rather than a single action's own legality. Capitals have no
such cap (buy_infantry there is limited only by silver/SPAWN_CAPS, not
even adjacency anymore - see the RULE CHANGE note above).

SCOPE: actions use the same atomic representation as v1 (hex identified
by index instead of coordinate) rather than the fixed/masked action-space
design discussed for the eventual RL policy - that design is tied to the
actual agent interface, which is a separate milestone from getting these
mechanics correct. Unit types are referenced by index here (0=infantry,
1=cavalry, 2=archers) matching state.UNIT_TYPES/SPAWN_CAPS order.
"""

import numpy as np

from .geometry import hex_distance
from .state import NO_FACTION, SPAWN_CAPS

INFANTRY_COST = 2
CAVALRY = 1
ARCHERS = 2

OUTPOST_COST = 3
OUTPOST_CAP = 6
OUTPOST_MIN_DIST_OWN_CAPITAL = 3    # "not within 2 tiles of your own capital"
OUTPOST_MIN_DIST_ENEMY_CAPITAL = 2  # "not within 1 tile of any other faction's capital"
OUTPOST_MIN_DIST_OTHER_OUTPOST = 2  # "not within 1 tile of any outpost" (yours or anyone else's)
UNIT_TYPE_INDEX = {"infantry": 0, "cavalry": 1, "archers": 2}


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


def _outpost_count(state, faction):
    return int(np.sum((state.city_owner == faction) & ~state.is_capital))


def _can_build_outpost(state, hex_index, faction):
    """Placement legality for a new outpost at hex_index: not on top of
    an existing capital/outpost, >= OUTPOST_MIN_DIST_OWN_CAPITAL from
    your own capital, >= OUTPOST_MIN_DIST_ENEMY_CAPITAL from any other
    faction's capital, and >= OUTPOST_MIN_DIST_OTHER_OUTPOST from any
    outpost at all (yours included - outposts can't be chained close
    together)."""
    if state.city_owner[hex_index] != NO_FACTION:
        return False

    grid = state.grid
    coord = grid.coord_of(hex_index)

    own_capital = np.nonzero((state.city_owner == faction) & state.is_capital)[0]
    if len(own_capital) and hex_distance(coord, grid.coord_of(int(own_capital[0]))) < OUTPOST_MIN_DIST_OWN_CAPITAL:
        return False

    enemy_capitals = np.nonzero(state.is_capital & (state.city_owner != NO_FACTION) & (state.city_owner != faction))[0]
    for c in enemy_capitals:
        if hex_distance(coord, grid.coord_of(int(c))) < OUTPOST_MIN_DIST_ENEMY_CAPITAL:
            return False

    all_outposts = np.nonzero((state.city_owner != NO_FACTION) & ~state.is_capital)[0]
    for o in all_outposts:
        if hex_distance(coord, grid.coord_of(int(o))) < OUTPOST_MIN_DIST_OTHER_OUTPOST:
            return False

    return True


def get_legal_buy_actions(state, faction):
    counts = count_all_units_in_play(state, faction)
    actions = []

    if _remaining_cap(counts, 0) > 0 and state.silver[faction] >= INFANTRY_COST:
        city_hexes = np.nonzero((state.city_owner == faction) & ~state.locked)[0]
        for hex_index in city_hexes:
            hex_index = int(hex_index)
            if state.is_capital[hex_index] or not _adjacent_enemy_present(state, hex_index, faction):
                actions.append({"type": "buy_infantry", "city_hex": hex_index})

    if state.kill_xp[faction] > 0 and state.silver[faction] >= 1:
        army_hexes = np.nonzero((state.army_faction == faction) & (state.army_units[:, 0] > 0))[0]
        for hex_index in army_hexes:
            hex_index = int(hex_index)
            for unit_index, unit_name in ((CAVALRY, "cavalry"), (ARCHERS, "archers")):
                if _remaining_cap(counts, unit_index) > 0:
                    actions.append({"type": "convert_to_special", "hex": hex_index, "unit_type": unit_name})

    if state.silver[faction] >= OUTPOST_COST and _outpost_count(state, faction) < OUTPOST_CAP:
        army_hexes = np.nonzero((state.army_faction == faction) & ~state.locked)[0]
        for hex_index in army_hexes:
            hex_index = int(hex_index)
            if not _can_build_outpost(state, hex_index, faction):
                continue
            for unit_index, unit_name in enumerate(("infantry", "cavalry", "archers")):
                if state.army_units[hex_index, unit_index] > 0:
                    actions.append({"type": "build_outpost", "hex": hex_index, "unit_type": unit_name})

    return actions


def _apply_one(state, faction, action, counts, enemy_adjacent_cache):
    if action["type"] == "buy_infantry":
        hex_index = action["city_hex"]
        if state.city_owner[hex_index] != faction or state.locked[hex_index]:
            return False

        if not state.is_capital[hex_index]:
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
        # city_owner==faction check above - a hostile army can't peacefully
        # sit on your capital/outpost anymore either (arriving there always
        # starts a battle now regardless of whether it's defended - see
        # movement.py) - but kept for exact behavioral parity.
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

    elif action["type"] == "build_outpost":
        hex_index = action["hex"]
        unit_index = UNIT_TYPE_INDEX[action["unit_type"]]
        if state.army_faction[hex_index] != faction or state.locked[hex_index]:
            return False
        if state.army_units[hex_index, unit_index] <= 0:
            return False
        if state.silver[faction] < OUTPOST_COST or _outpost_count(state, faction) >= OUTPOST_CAP:
            return False
        if not _can_build_outpost(state, hex_index, faction):
            return False

        state.silver[faction] -= OUTPOST_COST
        state.army_units[hex_index, unit_index] -= 1
        counts[unit_index] -= 1
        if int(state.army_units[hex_index].sum()) == 0:
            state.army_faction[hex_index] = NO_FACTION
        state.city_owner[hex_index] = faction
        return True

    return False


def apply_buy_phase(state, actions_by_faction):
    """actions_by_faction: {faction: [action, ...]}. Same caching pattern
    as engine/buy.py: one count snapshot + one enemy-adjacency cache per
    faction, maintained locally rather than rescanned per action. Also
    tracks, per faction, which outpost hexes have already recruited a
    unit this turn (buy_infantry there is capped at 1/turn - capitals
    have no such cap)."""
    for faction, actions in actions_by_faction.items():
        counts = count_all_units_in_play(state, faction)
        enemy_adjacent_cache = {}
        outpost_recruited = set()
        for action in actions:
            if action["type"] == "buy_infantry":
                hex_index = action["city_hex"]
                if (state.city_owner[hex_index] == faction and not state.is_capital[hex_index]
                        and hex_index in outpost_recruited):
                    continue
            ok = _apply_one(state, faction, action, counts, enemy_adjacent_cache)
            if ok and action["type"] == "buy_infantry":
                hex_index = action["city_hex"]
                if state.city_owner[hex_index] == faction and not state.is_capital[hex_index]:
                    outpost_recruited.add(hex_index)
    return state
