"""
Buy phase for engine - ported from engine/buy.py, plus outpost building and
upgrading for the outposts/VP/resources ruleset.

Four kinds of purchases, atomic (one unit each): buy_infantry (spend 2 gold
at an owned city - capital or outpost, both use city_owner the same way),
convert_to_special (spend 1 kill-XP + 1 gold to convert an existing
infantry unit into cavalry/archers), build_outpost (spend 3 gold + consume
1 unit already standing on the target hex to found a new outpost there),
and upgrade_outpost (spend that upgrade's resource cost - see
UPGRADE_COSTS - to give an owned outpost a Barracks/Workshop/Temple, or to
convert it directly from one upgrade to another at that upgrade's full
cost, no partial credit for the one being replaced). buy_infantry/
convert_to_special are subject to the 24/12/12 concurrent SPAWN_CAPS;
build_outpost is subject to OUTPOST_CAP instead (how many outposts one
faction may have standing at once) plus the placement-distance rules in
_can_build_outpost; upgrade_outpost only needs an owned, non-capital,
unlocked hex not already holding the requested upgrade.

RULE CHANGE - siege: buy_infantry's undefended-by-adjacent-enemy
requirement only applies at an outpost now, not a capital - a capital
can always recruit infantry regardless of what's adjacent to it (see
_adjacent_enemy_present's call sites below, both now gated on
`not is_capital`).

Recruiting at an outpost (buy_infantry there) is capped at 1 per turn
UNLESS that outpost has a Barracks upgrade, which removes the cap entirely
(the adjacent-enemy restriction still applies) - enforced in
apply_buy_phase, not here, since it's a per-turn-batch property rather
than a single action's own legality. Capitals have no such cap regardless
(buy_infantry there is limited only by gold/SPAWN_CAPS, not even adjacency -
see the RULE CHANGE note above).

Outpost actions (build_outpost and upgrade_outpost, combined) are capped
at 1 per turn per faction - also enforced in apply_buy_phase for the same
per-turn-batch reason, replacing the old unlimited-per-buy-phase building.

SCOPE: actions use the same atomic representation as v1 (hex identified
by index instead of coordinate) rather than the fixed/masked action-space
design discussed for the eventual RL policy - that design is tied to the
actual agent interface, which is a separate milestone from getting these
mechanics correct. Unit types are referenced by index here (0=infantry,
1=cavalry, 2=archers) matching state.UNIT_TYPES/SPAWN_CAPS order.
"""

import numpy as np

from .geometry import min_hex_distance_to_any
from .state import NO_FACTION, RESOURCE_TO_INDEX, SPAWN_CAPS, UPGRADE_TO_INDEX

INFANTRY_COST = 2
CAVALRY = 1
ARCHERS = 2

OUTPOST_COST = 3
OUTPOST_CAP = 6
OUTPOST_MIN_DIST_OWN_CAPITAL = 3    # "not within 2 tiles of your own capital"
OUTPOST_MIN_DIST_ENEMY_CAPITAL = 2  # "not within 1 tile of any other faction's capital"
OUTPOST_MIN_DIST_OTHER_OUTPOST = 2  # "not within 1 tile of any outpost" (yours or anyone else's)
UNIT_TYPE_INDEX = {"infantry": 0, "cavalry": 1, "archers": 2}

BARRACKS_INDEX = UPGRADE_TO_INDEX["barracks"]

# Resource cost to give an outpost each upgrade (or to convert it directly
# from a different upgrade - full cost, no credit for the one replaced).
UPGRADE_COSTS = {
    "barracks": {"fish": 2, "wood": 4},
    "workshop": {"iron": 2, "clay": 2, "wood": 4},
    "temple": {"fish": 2, "iron": 2, "clay": 2, "wood": 4},
}


def _can_afford_resources(state, faction, cost):
    return all(state.resources[faction, RESOURCE_TO_INDEX[r]] >= amount for r, amount in cost.items())


def _spend_resources(state, faction, cost):
    for r, amount in cost.items():
        state.resources[faction, RESOURCE_TO_INDEX[r]] -= amount


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


def eligible_outpost_mask(state, faction):
    """[num_hexes] bool - the same legality rules as _can_build_outpost
    (not on top of an existing capital/outpost, >= OUTPOST_MIN_DIST_OWN
    _CAPITAL from your own capital, >= OUTPOST_MIN_DIST_ENEMY_CAPITAL
    from any other faction's capital, >= OUTPOST_MIN_DIST_OTHER_OUTPOST
    from any outpost at all), computed for EVERY hex in one vectorized
    pass instead of one hex at a time.

    Added because _can_build_outpost used to be the dominant cost of a
    whole game: every agent's expansion-target scan (greedy_agent's
    _home_expansion_target, heuristic_agent's _best_expansion_target,
    vanguard_agent's _ranked_expansion_targets - the last one also
    inherited by marshal/warlord/sentinel/tactician) calls it once per
    CANDIDATE HEX, every movement/cavalry step, and it internally looped
    over every existing outpost computing hex_distance in Python for
    each candidate - O(hexes x outposts) in pure Python, redone from
    scratch every step even though almost nothing about outpost
    placement changes between consecutive steps. Profiled at 88% of
    total runtime in a 15-game greedy-vs-greedy sample (1.06M calls,
    13M hex_distance calls beneath them). Callers that need to check
    MANY candidate hexes should call this once per decision and index
    into the result; _can_build_outpost (kept for the few single-hex
    call sites) is now defined in terms of this function."""
    grid = state.grid
    coords = grid.coords_array
    mask = state.city_owner == NO_FACTION

    own_capital = np.nonzero((state.city_owner == faction) & state.is_capital)[0]
    if len(own_capital):
        dist = min_hex_distance_to_any(coords, own_capital)
        mask = mask & (dist >= OUTPOST_MIN_DIST_OWN_CAPITAL)

    enemy_capitals = np.nonzero(state.is_capital & (state.city_owner != NO_FACTION) & (state.city_owner != faction))[0]
    if len(enemy_capitals):
        dist = min_hex_distance_to_any(coords, enemy_capitals)
        mask = mask & (dist >= OUTPOST_MIN_DIST_ENEMY_CAPITAL)

    all_outposts = np.nonzero((state.city_owner != NO_FACTION) & ~state.is_capital)[0]
    if len(all_outposts):
        dist = min_hex_distance_to_any(coords, all_outposts)
        mask = mask & (dist >= OUTPOST_MIN_DIST_OTHER_OUTPOST)

    return mask


def _can_build_outpost(state, hex_index, faction):
    """Placement legality for a new outpost at hex_index - see
    eligible_outpost_mask's docstring for the actual rules and for why
    that's the vectorized version of this same check. This single-hex
    form is for the few call sites that only ever need one hex (mainly
    _apply_one's build_outpost validation); anything scanning multiple
    candidate hexes should call eligible_outpost_mask directly instead
    of calling this in a loop."""
    return bool(eligible_outpost_mask(state, faction)[hex_index])


def get_legal_buy_actions(state, faction):
    counts = count_all_units_in_play(state, faction)
    actions = []

    if _remaining_cap(counts, 0) > 0 and state.gold[faction] >= INFANTRY_COST:
        city_hexes = np.nonzero((state.city_owner == faction) & ~state.locked)[0]
        for hex_index in city_hexes:
            hex_index = int(hex_index)
            if state.is_capital[hex_index] or not _adjacent_enemy_present(state, hex_index, faction):
                actions.append({"type": "buy_infantry", "city_hex": hex_index})

    if state.kill_xp[faction] > 0 and state.gold[faction] >= 1:
        army_hexes = np.nonzero((state.army_faction == faction) & (state.army_units[:, 0] > 0))[0]
        for hex_index in army_hexes:
            hex_index = int(hex_index)
            for unit_index, unit_name in ((CAVALRY, "cavalry"), (ARCHERS, "archers")):
                if _remaining_cap(counts, unit_index) > 0:
                    actions.append({"type": "convert_to_special", "hex": hex_index, "unit_type": unit_name})

    if state.gold[faction] >= OUTPOST_COST and _outpost_count(state, faction) < OUTPOST_CAP:
        army_hexes = np.nonzero((state.army_faction == faction) & ~state.locked)[0]
        eligible = eligible_outpost_mask(state, faction)
        for hex_index in army_hexes:
            hex_index = int(hex_index)
            if not eligible[hex_index]:
                continue
            for unit_index, unit_name in enumerate(("infantry", "cavalry", "archers")):
                if state.army_units[hex_index, unit_index] > 0:
                    actions.append({"type": "build_outpost", "hex": hex_index, "unit_type": unit_name})

    outpost_hexes = np.nonzero((state.city_owner == faction) & ~state.is_capital & ~state.locked)[0]
    for hex_index in outpost_hexes:
        hex_index = int(hex_index)
        current = int(state.outpost_upgrade[hex_index])
        for upgrade, cost in UPGRADE_COSTS.items():
            if UPGRADE_TO_INDEX[upgrade] == current:
                continue
            if _can_afford_resources(state, faction, cost):
                actions.append({"type": "upgrade_outpost", "hex": hex_index, "upgrade": upgrade})

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

        if state.gold[faction] < INFANTRY_COST or _remaining_cap(counts, 0) <= 0:
            return False

        # Stack-cap check moved BEFORE the gold deduction below (it used to
        # run after spending the gold, so a purchase that failed here still
        # cost the faction 2 gold for nothing - confirmed to waste 38 of a
        # faction's starting 50 gold on turn 1 alone, since greedy_buy
        # proposes gold // INFANTRY_COST purchases at one hex with no
        # awareness of how many actually fit - see heuristic_agent.py's
        # docstring for the prior agent-side attempt at this fix).
        # An empty hex (NO_FACTION - e.g. a freshly-built outpost with no
        # army on it yet) is fine, we'd be the first army there; anything
        # else not already ours can't actually trigger given the
        # city_owner==faction check above (a hostile army can't peacefully
        # sit on your capital/outpost - arriving there always starts a
        # battle now, see movement.py) but kept for parity.
        if state.army_faction[hex_index] not in (NO_FACTION, faction) or int(state.army_units[hex_index].sum()) >= 6:
            return False

        state.gold[faction] -= INFANTRY_COST
        if state.army_faction[hex_index] == NO_FACTION:
            state.army_faction[hex_index] = faction
        state.army_units[hex_index, 0] += 1
        counts[0] += 1
        return True

    elif action["type"] == "convert_to_special":
        hex_index = action["hex"]
        unit_index = CAVALRY if action["unit_type"] == "cavalry" else ARCHERS
        if state.army_faction[hex_index] != faction or state.army_units[hex_index, 0] <= 0:
            return False
        if state.kill_xp[faction] <= 0 or state.gold[faction] < 1 or _remaining_cap(counts, unit_index) <= 0:
            return False

        state.kill_xp[faction] -= 1
        state.gold[faction] -= 1
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
        if state.gold[faction] < OUTPOST_COST or _outpost_count(state, faction) >= OUTPOST_CAP:
            return False
        if not _can_build_outpost(state, hex_index, faction):
            return False

        state.gold[faction] -= OUTPOST_COST
        state.army_units[hex_index, unit_index] -= 1
        counts[unit_index] -= 1
        if int(state.army_units[hex_index].sum()) == 0:
            state.army_faction[hex_index] = NO_FACTION
        state.city_owner[hex_index] = faction
        return True

    elif action["type"] == "upgrade_outpost":
        hex_index = action["hex"]
        upgrade = action["upgrade"]
        if state.city_owner[hex_index] != faction or state.is_capital[hex_index] or state.locked[hex_index]:
            return False
        upgrade_index = UPGRADE_TO_INDEX[upgrade]
        if state.outpost_upgrade[hex_index] == upgrade_index:
            return False
        cost = UPGRADE_COSTS[upgrade]
        if not _can_afford_resources(state, faction, cost):
            return False

        _spend_resources(state, faction, cost)
        state.outpost_upgrade[hex_index] = upgrade_index
        return True

    return False


def apply_buy_phase(state, actions_by_faction):
    """actions_by_faction: {faction: [action, ...]}. Same caching pattern
    as engine/buy.py: one count snapshot + one enemy-adjacency cache per
    faction, maintained locally rather than rescanned per action. Also
    tracks, per faction:
      - which outpost hexes have already recruited a unit this turn
        (buy_infantry there is capped at 1/turn, UNLESS that outpost has a
        Barracks - capitals never have this cap either way);
      - whether an outpost action (build_outpost or upgrade_outpost,
        combined) has already been taken this turn - capped at 1/turn.
    Both caps are per-turn-BATCH properties (which action(s) in this same
    call already happened), not a single action's own legality, so they're
    enforced here rather than in get_legal_buy_actions - same reasoning as
    engine/buy.py."""
    for faction, actions in actions_by_faction.items():
        counts = count_all_units_in_play(state, faction)
        enemy_adjacent_cache = {}
        outpost_recruited = set()
        outpost_action_used = False
        for action in actions:
            if action["type"] == "buy_infantry":
                hex_index = action["city_hex"]
                if (state.city_owner[hex_index] == faction and not state.is_capital[hex_index]
                        and state.outpost_upgrade[hex_index] != BARRACKS_INDEX
                        and hex_index in outpost_recruited):
                    continue
            if action["type"] in ("build_outpost", "upgrade_outpost") and outpost_action_used:
                continue

            ok = _apply_one(state, faction, action, counts, enemy_adjacent_cache)
            if not ok:
                continue
            if action["type"] == "buy_infantry":
                hex_index = action["city_hex"]
                if state.city_owner[hex_index] == faction and not state.is_capital[hex_index]:
                    outpost_recruited.add(hex_index)
            elif action["type"] in ("build_outpost", "upgrade_outpost"):
                outpost_action_used = True
    return state
