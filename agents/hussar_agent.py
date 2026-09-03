"""
HussarAgent for engine: vanguard_agent's army-cycling movement plus a
buy-phase change aimed at a resource vanguard_agent itself doesn't touch:
the cavalry movement phase's 2 extra per-turn move slots (on top of the
regular phase's 3) are only usable by hexes that currently have at least
one cavalry unit standing on them - a faction with zero cavalry gets ZERO
benefit from that phase, every single turn, no matter how many infantry/
archers it owns.

HussarAgent's buy phase (hussar_buy) is greedy_buy's, with one change:
kill-XP conversions always prefer cavalry over archers while there's room
under its 12-unit cap, only converting to archers once cavalry is maxed
out. Combined with vanguard_move's rotation, more cavalry-bearing hexes
means more of a turn's 5 move slots actually have something eligible to
use them.

RESULT: inconclusive - see this module's git history/prior test notes.
vs plain greedy_buy+greedy movement this tests as strong as
vanguard_agent itself, but head-to-head against plain vanguard_agent
(identical except for this cavalry bias) came out dead on the no-effect
baseline across two independent samples. Kept in the package as
genuinely tied with vanguard_agent, not worse.
"""

import random

from .greedy_agent import (
    _UPGRADE_PRIORITY, _can_afford_upgrade, greedy_draft, greedy_placement, greedy_resource_choice, greedy_swap,
)
from .heuristic_agent import heuristic_target
from .random_agent import random_rectification
from .vanguard_agent import vanguard_move
from engine.buy import OUTPOST_CAP, eligible_outpost_mask, _outpost_count
from engine.state import NO_UPGRADE, SPAWN_CAPS, count_units_in_play

_OUTPOST_UNIT_PRIORITY = (0, 1, 2)  # infantry, cavalry, archers


def hussar_buy(state, faction, rng):
    """Identical to greedy_agent's greedy_buy except the kill-XP
    conversion loop always prefers cavalry (up to its SPAWN_CAP) - see
    module docstring."""
    decision = {}

    outposts = ((state.city_owner[0] == faction) & ~state.is_capital[0] & ~state.locked[0]).nonzero(as_tuple=False).flatten().tolist()
    upgradeable = [
        h for h in outposts
        if int(state.outpost_upgrade[0, h]) == NO_UPGRADE and any(_can_afford_upgrade(state, faction, u) for u in _UPGRADE_PRIORITY)
    ]

    if upgradeable:
        h = rng.choice(upgradeable)
        for upgrade in _UPGRADE_PRIORITY:
            if _can_afford_upgrade(state, faction, upgrade):
                decision["outpost_type"] = 2
                decision["outpost_hex"] = h
                decision["outpost_upgrade"] = upgrade
                break
    else:
        army_hexes = ((state.army_faction[0] == faction) & ~state.locked[0]).nonzero(as_tuple=False).flatten().tolist()
        if army_hexes and int(state.gold[0, faction]) >= 3 and int(_outpost_count(state, faction)[0]) < OUTPOST_CAP:
            eligible = eligible_outpost_mask(state, faction)[0]
            candidates = [h for h in army_hexes if bool(eligible[h])]
            rng.shuffle(candidates)
            for h in candidates:
                for unit_type in _OUTPOST_UNIT_PRIORITY:
                    if int(state.army_units[0, h, unit_type]) > 0:
                        decision["outpost_type"] = 1
                        decision["outpost_hex"] = h
                        decision["outpost_unit_type"] = unit_type
                        break
                if "outpost_type" in decision:
                    break

    city_hexes = ((state.city_owner[0] == faction) & ~state.locked[0]).nonzero(as_tuple=False).flatten().tolist()
    if city_hexes and int(state.gold[0, faction]) >= 2:
        h = rng.choice(city_hexes)
        decision["infantry_buy"] = {h: int(state.gold[0, faction]) // 2}

    army_hexes = ((state.army_faction[0] == faction) & (state.army_units[0, :, 0] > 0)).nonzero(as_tuple=False).flatten().tolist()
    kill_xp = int(state.kill_xp[0, faction])
    if army_hexes and kill_xp > 0:
        cav_count = int(count_units_in_play(state, faction, 1)[0])
        convert_cavalry, convert_archers = {}, {}
        for _ in range(kill_xp):
            h = rng.choice(army_hexes)
            if cav_count < int(SPAWN_CAPS[1]):
                convert_cavalry[h] = convert_cavalry.get(h, 0) + 1
                cav_count += 1
            else:
                convert_archers[h] = convert_archers.get(h, 0) + 1
        if convert_cavalry:
            decision["convert_cavalry"] = convert_cavalry
        if convert_archers:
            decision["convert_archers"] = convert_archers

    return decision


def make_hussar_agents(num_factions, seed=0):
    """Same 9-callback shape as the other make_X_agents. Rectification/
    resource-choice/placement/draft/swap match greedy_agent's; buy is
    hussar_buy, movement is vanguard_agent's vanguard_move."""
    rngs = {f: random.Random(seed * 1_000_003 + f) for f in range(num_factions)}

    def decide_buy(state, faction):
        return hussar_buy(state, faction, rngs[faction])

    def decide_movement(state, faction, step, legal_mask):
        return vanguard_move(state, faction, step, legal_mask)

    def decide_cavalry(state, faction, step, legal_mask):
        return vanguard_move(state, faction, step, legal_mask)

    def decide_target(state, hex_index, faction):
        return heuristic_target(state, hex_index, faction)

    def decide_rectification(state, hex_index, winner_faction, cap):
        return random_rectification(state, hex_index, winner_faction, cap, rngs[winner_faction])

    def decide_resource_choice(state, faction, hex_index):
        return greedy_resource_choice(state, faction)

    def decide_placement(state, faction, legal_mask):
        return greedy_placement(state, legal_mask)

    def decide_draft(state, faction, legal_pool):
        return greedy_draft(state, legal_pool)

    def decide_swap(state, faction, leftover_hex, placer_faction, placer_hex):
        return greedy_swap(state, leftover_hex, placer_hex)

    factions = range(num_factions)
    return (
        {f: decide_buy for f in factions},
        {f: decide_movement for f in factions},
        {f: decide_cavalry for f in factions},
        {f: decide_target for f in factions},
        {f: decide_rectification for f in factions},
        {f: decide_resource_choice for f in factions},
        {f: decide_placement for f in factions},
        {f: decide_draft for f in factions},
        {f: decide_swap for f in factions},
    )
