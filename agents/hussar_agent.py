"""
HussarAgent for engine: vanguard_agent's army-cycling movement (see that
module's docstring) plus a buy-phase change aimed at a resource vanguard_
agent itself doesn't touch: the cavalry movement phase's 2 extra
per-turn move slots (on top of the regular phase's 3) are only usable by
hexes that currently have at least one cavalry unit standing on them
(see engine/movement.py's legal_cavalry_mask) - a faction with zero
cavalry gets ZERO benefit from that phase, every single turn, no matter
how many infantry/archers it owns. Early game especially, before any
kill-XP conversions happen, this phase is dead weight for everyone.

HussarAgent's buy phase (hussar_buy) is greedy_buy's, with one change:
kill-XP conversions always prefer cavalry over archers (greedy_buy's own
tie-break already leans that way when counts are equal, but keeps
splitting toward whichever is currently behind once cavalry gets ahead -
this version just always takes cavalry while there's room under its
12-unit cap, only converting to archers once cavalry is maxed out).
Combined with vanguard_move's rotation, more cavalry-bearing hexes means
more of a turn's 5 move slots actually have something eligible to use
them, not just the regular phase's 3.

RESULT: inconclusive. vs plain greedy_buy+greedy movement this tests as
strong as vanguard_agent itself (~28% win rate, matching that module's
own numbers), but two independent 150-game samples head-to-head against
plain vanguard_agent (identical except for this cavalry bias) came out
at 14.7% and 12.0% - averaging almost exactly the ~12.5% baseline, no
real signal either way. Best guess: converting a unit to cavalry costs
1 kill-XP (a scarce resource, only earned through combat) AND removes
an infantry unit, so activating extra cavalry-phase move slots this way
trades off against overall army mass - and kill-XP is especially scarce
early, when the extra slots would matter most. Kept in the package
rather than reverted (it's genuinely tied with vanguard_agent, not
worse), but don't read the vs-greedy number as evidence this bias
itself is doing anything - see vanguard_agent for that credit.
"""

import random

from .greedy_agent import greedy_draft, greedy_placement, greedy_resource_choice, greedy_swap
from .heuristic_agent import heuristic_target
from .random_agent import random_rectification
from .vanguard_agent import vanguard_move
from engine.buy import INFANTRY_COST
from engine.state import NO_UPGRADE, SPAWN_CAPS, count_units_in_play

_UPGRADE_PRIORITY = ("temple", "barracks", "workshop")
_OUTPOST_UNIT_PRIORITY = ("infantry", "cavalry", "archers")


def hussar_buy(state, faction, legal, rng):
    """Identical to greedy_agent's greedy_buy except the kill-XP
    conversion loop always prefers cavalry - see module docstring."""
    outpost_actions = [a for a in legal if a["type"] == "build_outpost"]
    upgrade_actions = [
        a for a in legal
        if a["type"] == "upgrade_outpost" and state.outpost_upgrade[a["hex"]] == NO_UPGRADE
    ]
    infantry_actions = [a for a in legal if a["type"] == "buy_infantry"]
    convert_actions = [a for a in legal if a["type"] == "convert_to_special"]

    chosen = []

    if upgrade_actions:
        by_hex = {}
        for a in upgrade_actions:
            by_hex.setdefault(a["hex"], {})[a["upgrade"]] = a
        upgrade_hexes = list(by_hex.keys())
        rng.shuffle(upgrade_hexes)
        options = by_hex[upgrade_hexes[0]]
        for upgrade in _UPGRADE_PRIORITY:
            if upgrade in options:
                chosen.append(options[upgrade])
                break
    else:
        by_hex = {}
        for a in outpost_actions:
            by_hex.setdefault(a["hex"], {})[a["unit_type"]] = a
        outpost_hexes = list(by_hex.keys())
        rng.shuffle(outpost_hexes)
        if outpost_hexes:
            options = by_hex[outpost_hexes[0]]
            for unit_type in _OUTPOST_UNIT_PRIORITY:
                if unit_type in options:
                    chosen.append(options[unit_type])
                    break

    if infantry_actions:
        num_purchases = int(state.gold[faction]) // INFANTRY_COST
        for _ in range(num_purchases):
            chosen.append(rng.choice(infantry_actions))

    if convert_actions:
        num_conversions = int(state.kill_xp[faction])
        cav_count = count_units_in_play(state, faction, 1)
        for _ in range(num_conversions):
            unit_type = "cavalry" if cav_count < int(SPAWN_CAPS[1]) else "archers"
            matching = [a for a in convert_actions if a["unit_type"] == unit_type]
            if not matching:
                matching = convert_actions
            if matching:
                chosen.append(rng.choice(matching))
                if unit_type == "cavalry":
                    cav_count += 1

    return chosen


def make_hussar_agents(num_factions, seed=0):
    """Same 9-callback shape as the other make_X_agents. Rectification/
    resource-choice/placement/draft/swap match greedy_agent's; buy is
    hussar_buy, movement is vanguard_agent's vanguard_move, target is
    heuristic_agent's heuristic_target (see module docstring)."""
    rngs = {f: random.Random(seed * 1_000_003 + f) for f in range(num_factions)}

    def decide_buy(state, faction, legal):
        return hussar_buy(state, faction, legal, rngs[faction])

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
