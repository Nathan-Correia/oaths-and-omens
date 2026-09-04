"""
Hand-built buy-phase scenarios for the per-turn batch rules.

WHY: mutation-testing M3 found that removing the "one outpost action per turn"
cap entirely broke NOTHING in 180 turns of traced play or 27 movement scenarios.
The rule is only observable when an agent proposes two outpost actions in the same
turn, and no traced agent ever does. Same shape of hole for the per-outpost recruit
cap, the Barracks exemption, and the adjacency and placement rules.

These scenarios propose action lists that real agents do not, precisely so the
batch rules are forced to bite.

Usage:  python engine/tools/dump_buy_scenarios.py > engine/tests/data/buy_scenarios.txt
"""

import copy
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _bootstrap  # noqa: E402

_bootstrap.install()

from engine.buy import apply_buy_phase  # noqa: E402
from engine.setup import create_initial_state  # noqa: E402
from engine.state import NO_FACTION  # noqa: E402

from state_io import write_state  # noqa: E402

RADIUS = 7
NUM_FACTIONS = 4
UNIT_INDEX = {"infantry": 0, "cavalry": 1, "archers": 2}
UPGRADE_INDEX = {"barracks": 0, "workshop": 1, "temple": 2}
BUY_TYPE_INDEX = {
    "buy_infantry": 0,
    "convert_to_special": 1,
    "build_outpost": 2,
    "upgrade_outpost": 3,
}


def blank_state(seed=11):
    s = create_initial_state(radius=RADIUS, num_factions=NUM_FACTIONS, seed=seed)
    s.terrain[:] = 0  # all plains, so placement is never blocked by terrain
    s.city_owner[:] = NO_FACTION
    s.is_capital[:] = False
    s.city_placer[:] = NO_FACTION
    s.army_faction[:] = NO_FACTION
    s.army_units[:] = 0
    s.outpost_upgrade[:] = -1
    s.frozen[:] = False
    s.locked[:] = False
    s.gold[:] = 0
    s.kill_xp[:] = 0
    s.resources[:] = 0
    return s


def put(s, hex_index, faction, inf=0, cav=0, arc=0):
    s.army_faction[hex_index] = faction
    s.army_units[hex_index] = [inf, cav, arc]


def far_apart(s, count, min_dist=4):
    """`count` hexes pairwise at least min_dist apart, so outpost placement rules
    never interfere unless a scenario wants them to. Pass min_dist=1 when a
    scenario only needs distinct hexes and spacing is irrelevant - a radius-7
    board cannot supply many widely separated hexes."""
    from engine.geometry import hex_distance
    chosen = []
    for h in range(s.num_hexes):
        c = s.grid.coord_of(h)
        if all(hex_distance(c, s.grid.coord_of(o)) >= min_dist for o in chosen):
            chosen.append(h)
            if len(chosen) == count:
                return chosen
    raise RuntimeError("not enough spread-out hexes")


SCENARIOS = []


def scenario(name):
    def deco(fn):
        SCENARIOS.append((name, fn))
        return fn
    return deco


# --- the per-turn batch caps (the hole mutation testing found) ----------------

@scenario("two_build_outposts_one_turn")
def _(s):
    a, b = far_apart(s, 2)
    put(s, a, 0, inf=2)
    put(s, b, 0, inf=2)
    s.gold[0] = 50
    return {0: [{"type": "build_outpost", "hex": a, "unit_type": "infantry"},
                {"type": "build_outpost", "hex": b, "unit_type": "infantry"}]}


@scenario("build_then_upgrade_one_turn")
def _(s):
    a, b = far_apart(s, 2)
    put(s, a, 0, inf=2)
    s.city_owner[b] = 0
    s.gold[0] = 50
    s.resources[0] = [10, 10, 10, 10]
    return {0: [{"type": "build_outpost", "hex": a, "unit_type": "infantry"},
                {"type": "upgrade_outpost", "hex": b, "upgrade": "barracks"}]}


@scenario("two_upgrades_one_turn")
def _(s):
    a, b = far_apart(s, 2)
    s.city_owner[a] = 0
    s.city_owner[b] = 0
    s.resources[0] = [20, 20, 20, 20]
    return {0: [{"type": "upgrade_outpost", "hex": a, "upgrade": "barracks"},
                {"type": "upgrade_outpost", "hex": b, "upgrade": "temple"}]}


@scenario("upgrade_conversion_costs_full_price")
def _(s):
    a = far_apart(s, 1)[0]
    s.city_owner[a] = 0
    s.outpost_upgrade[a] = UPGRADE_INDEX["barracks"]
    s.resources[0] = [4, 2, 2, 2]  # exactly a temple
    return {0: [{"type": "upgrade_outpost", "hex": a, "upgrade": "temple"}]}


@scenario("two_recruits_same_outpost_capped")
def _(s):
    a = far_apart(s, 1)[0]
    s.city_owner[a] = 0
    s.gold[0] = 50
    return {0: [{"type": "buy_infantry", "city_hex": a},
                {"type": "buy_infantry", "city_hex": a}]}


@scenario("two_recruits_barracks_outpost_uncapped")
def _(s):
    a = far_apart(s, 1)[0]
    s.city_owner[a] = 0
    s.outpost_upgrade[a] = UPGRADE_INDEX["barracks"]
    s.gold[0] = 50
    return {0: [{"type": "buy_infantry", "city_hex": a},
                {"type": "buy_infantry", "city_hex": a},
                {"type": "buy_infantry", "city_hex": a}]}


@scenario("many_recruits_at_capital_uncapped")
def _(s):
    a = far_apart(s, 1)[0]
    s.city_owner[a] = 0
    s.is_capital[a] = True
    s.gold[0] = 50
    return {0: [{"type": "buy_infantry", "city_hex": a} for _ in range(8)]}


# --- adjacency and the stack cap ---------------------------------------------

@scenario("outpost_recruit_blocked_by_adjacent_enemy")
def _(s):
    a = far_apart(s, 1)[0]
    s.city_owner[a] = 0
    nbr = int(s.grid.neighbor_table[a, 0])
    put(s, nbr, 1, inf=1)
    s.gold[0] = 50
    return {0: [{"type": "buy_infantry", "city_hex": a}]}


@scenario("capital_recruit_ignores_adjacent_enemy")
def _(s):
    a = far_apart(s, 1)[0]
    s.city_owner[a] = 0
    s.is_capital[a] = True
    nbr = int(s.grid.neighbor_table[a, 0])
    put(s, nbr, 1, inf=1)
    s.gold[0] = 50
    return {0: [{"type": "buy_infantry", "city_hex": a}]}


@scenario("recruit_blocked_by_full_stack")
def _(s):
    a = far_apart(s, 1)[0]
    s.city_owner[a] = 0
    s.is_capital[a] = True
    put(s, a, 0, inf=6)
    s.gold[0] = 50
    # Gold must NOT be spent on the refused purchase - engine_old used to lose it.
    return {0: [{"type": "buy_infantry", "city_hex": a} for _ in range(4)]}


@scenario("recruit_hits_spawn_cap")
def _(s):
    a, b = far_apart(s, 2)
    s.city_owner[a] = 0
    s.is_capital[a] = True
    put(s, b, 0, inf=6)
    # 23 already in play across several hexes, so the 24 cap bites mid-list.
    spread = far_apart(s, 6, min_dist=1)
    for i, h in enumerate(spread[2:]):
        put(s, h, 0, inf=6 if i < 2 else 5)
    s.gold[0] = 50
    return {0: [{"type": "buy_infantry", "city_hex": a} for _ in range(6)]}


# --- conversions --------------------------------------------------------------

@scenario("convert_to_cavalry_and_archers")
def _(s):
    a = far_apart(s, 1)[0]
    put(s, a, 0, inf=4)
    s.gold[0] = 10
    s.kill_xp[0] = 5
    return {0: [{"type": "convert_to_special", "hex": a, "unit_type": "cavalry"},
                {"type": "convert_to_special", "hex": a, "unit_type": "archers"}]}


@scenario("convert_without_kill_xp_refused")
def _(s):
    a = far_apart(s, 1)[0]
    put(s, a, 0, inf=4)
    s.gold[0] = 10
    s.kill_xp[0] = 0
    return {0: [{"type": "convert_to_special", "hex": a, "unit_type": "cavalry"}]}


@scenario("convert_hits_cavalry_cap")
def _(s):
    spread = far_apart(s, 4, min_dist=1)
    for h in spread:
        put(s, h, 0, inf=2, cav=3)
    s.gold[0] = 50
    s.kill_xp[0] = 50
    return {0: [{"type": "convert_to_special", "hex": spread[0], "unit_type": "cavalry"}
                for _ in range(4)]}


# --- outpost placement rules --------------------------------------------------

@scenario("build_too_close_to_own_capital")
def _(s):
    a = far_apart(s, 1)[0]
    s.city_owner[a] = 0
    s.is_capital[a] = True
    nbr = int(s.grid.neighbor_table[a, 0])
    two = int(s.grid.neighbor_table[nbr, 0])
    put(s, two, 0, inf=2)  # distance 2 from own capital, needs 3
    s.gold[0] = 50
    return {0: [{"type": "build_outpost", "hex": two, "unit_type": "infantry"}]}


@scenario("build_at_exactly_min_distance_from_own_capital")
def _(s):
    a = far_apart(s, 1)[0]
    s.city_owner[a] = 0
    s.is_capital[a] = True
    h = a
    for _ in range(3):
        h = int(s.grid.neighbor_table[h, 0])
    put(s, h, 0, inf=2)  # distance 3, the first legal ring
    s.gold[0] = 50
    return {0: [{"type": "build_outpost", "hex": h, "unit_type": "infantry"}]}


@scenario("build_adjacent_to_enemy_capital")
def _(s):
    a = far_apart(s, 1)[0]
    s.city_owner[a] = 1
    s.is_capital[a] = True
    nbr = int(s.grid.neighbor_table[a, 0])
    put(s, nbr, 0, inf=2)
    s.gold[0] = 50
    return {0: [{"type": "build_outpost", "hex": nbr, "unit_type": "infantry"}]}


@scenario("build_adjacent_to_existing_outpost")
def _(s):
    a = far_apart(s, 1)[0]
    s.city_owner[a] = 1
    nbr = int(s.grid.neighbor_table[a, 0])
    put(s, nbr, 0, inf=2)
    s.gold[0] = 50
    return {0: [{"type": "build_outpost", "hex": nbr, "unit_type": "infantry"}]}


@scenario("build_at_outpost_cap")
def _(s):
    spread = far_apart(s, 8)
    for h in spread[:6]:
        s.city_owner[h] = 0
    put(s, spread[7], 0, inf=2)
    s.gold[0] = 50
    return {0: [{"type": "build_outpost", "hex": spread[7], "unit_type": "infantry"}]}


@scenario("build_consumes_last_unit_clears_army")
def _(s):
    a = far_apart(s, 1)[0]
    put(s, a, 0, cav=1)
    s.gold[0] = 50
    return {0: [{"type": "build_outpost", "hex": a, "unit_type": "cavalry"}]}


# --- multi-faction, and the adjacency cache quirk ------------------------------

@scenario("two_factions_same_turn")
def _(s):
    a, b = far_apart(s, 2)
    s.city_owner[a] = 0
    s.is_capital[a] = True
    s.city_owner[b] = 1
    s.is_capital[b] = True
    s.gold[0] = 20
    s.gold[1] = 20
    return {0: [{"type": "buy_infantry", "city_hex": a}] * 3,
            1: [{"type": "buy_infantry", "city_hex": b}] * 2}


@scenario("adjacency_cache_is_stale_by_design")
def _(s):
    """engine_old computes adjacency once per hex per buy phase and keeps using it
    even after purchases change the board. Faction 0 recruits at a capital next to
    an outpost of its own; the cached 'no adjacent enemy' verdict for the outpost
    must survive whatever else happens this phase."""
    a = far_apart(s, 1)[0]
    s.city_owner[a] = 0
    s.is_capital[a] = True
    h = a
    for _ in range(3):
        h = int(s.grid.neighbor_table[h, 0])
    s.city_owner[h] = 0
    s.gold[0] = 50
    return {0: [{"type": "buy_infantry", "city_hex": h},
                {"type": "buy_infantry", "city_hex": a},
                {"type": "buy_infantry", "city_hex": h}]}


@scenario("locked_city_cannot_recruit")
def _(s):
    a = far_apart(s, 1)[0]
    s.city_owner[a] = 0
    s.is_capital[a] = True
    s.locked[a] = True
    s.battle_faction[a, 0] = 1
    s.battle_units[a, 0] = [3, 0, 0]
    s.battle_origin[a, 0] = a
    s.battle_order.append(a)
    s.gold[0] = 50
    return {0: [{"type": "buy_infantry", "city_hex": a}]}


def _clone(state):
    new = copy.copy(state)
    for field in ("terrain", "city_owner", "is_capital", "outpost_upgrade", "city_placer",
                  "capital_settle_order", "army_faction", "army_units", "frozen", "locked",
                  "battle_faction", "battle_origin", "battle_units", "battle_moved",
                  "battle_round", "gold", "resources", "kill_xp", "victory_points", "alive"):
        setattr(new, field, getattr(state, field).copy())
    new.battle_order = list(state.battle_order)
    return new


def action_tokens(action):
    kind = BUY_TYPE_INDEX[action["type"]]
    if action["type"] == "buy_infantry":
        return f"{kind} {action['city_hex']} 0 0"
    if action["type"] == "convert_to_special":
        return f"{kind} {action['hex']} {UNIT_INDEX[action['unit_type']]} 0"
    if action["type"] == "build_outpost":
        return f"{kind} {action['hex']} {UNIT_INDEX[action['unit_type']]} 0"
    return f"{kind} {action['hex']} 0 {UPGRADE_INDEX[action['upgrade']]}"


def main():
    out = sys.stdout
    cases = []
    for name, build in SCENARIOS:
        s = blank_state()
        actions = build(s)
        before = _clone(s)
        after = _clone(s)
        apply_buy_phase(after, actions)
        cases.append((name, actions, before, after))

    out.write(f"BUY_SCENARIOS {len(cases)}\n")
    for name, actions, before, after in cases:
        out.write(f"BUY_SCENARIO {name}\n")
        out.write(f"FACTIONS {len(actions)}\n")
        for faction, alist in actions.items():
            toks = " ".join(action_tokens(a) for a in alist)
            out.write(f"{faction} {len(alist)}" + (" " + toks if toks else "") + "\n")
        write_state(out, before)
        write_state(out, after)


if __name__ == "__main__":
    main()
