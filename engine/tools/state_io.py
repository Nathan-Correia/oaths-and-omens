"""
Writes engine_old's ArrayState in the canonical text format src/state_io.cpp reads.

The two implementations must agree exactly; keep them in step. See PLAN.md §3.3 -
this is how a C++ phase gets compared against a Python-produced before/after pair.

Battle contributions are written sparsely (occupied slots only), since a dense
[num_hexes][MAX_BATTLE_CONTRIB] dump would be almost all padding.
"""

import numpy as np


def _arr(out, name, values):
    values = list(values)
    out.write(f"{name} {len(values)}")
    for v in values:
        out.write(f" {int(v)}")
    out.write("\n")


def write_state(out, state):
    n = int(state.num_hexes)
    f = int(state.num_factions)

    out.write("STATE 1\n")
    out.write(f"RADIUS {int(state.grid.radius)}\n")
    out.write(f"NUM_FACTIONS {f}\n")
    out.write(f"NUM_HEXES {n}\n")
    out.write(f"TURN {int(state.turn_number)}\n")

    _arr(out, "TERRAIN", state.terrain[:n])
    _arr(out, "CITY_OWNER", state.city_owner[:n])
    _arr(out, "IS_CAPITAL", state.is_capital[:n].astype(np.int32))
    _arr(out, "OUTPOST_UPGRADE", state.outpost_upgrade[:n])
    _arr(out, "CITY_PLACER", state.city_placer[:n])
    _arr(out, "ARMY_FACTION", state.army_faction[:n])
    _arr(out, "ARMY_UNITS", state.army_units[:n].reshape(-1))
    _arr(out, "FROZEN", state.frozen[:n].astype(np.int32))
    _arr(out, "LOCKED", state.locked[:n].astype(np.int32))
    _arr(out, "BATTLE_ROUND", state.battle_round[:n])

    slots = []
    k_max = state.battle_faction.shape[1]
    for h in range(n):
        for k in range(k_max):
            if int(state.battle_faction[h, k]) == -1:
                continue
            slots.append((
                h, k,
                int(state.battle_faction[h, k]),
                int(state.battle_origin[h, k]),
                int(state.battle_units[h, k, 0]),
                int(state.battle_units[h, k, 1]),
                int(state.battle_units[h, k, 2]),
                1 if bool(state.battle_moved[h, k]) else 0,
            ))
    out.write(f"BATTLE_SLOTS {len(slots)}\n")
    for row in slots:
        out.write(" ".join(str(x) for x in row) + "\n")

    _arr(out, "BATTLE_ORDER", state.battle_order)
    _arr(out, "CAPITAL_SETTLE_ORDER", state.capital_settle_order[:f])
    _arr(out, "GOLD", state.gold[:f])
    _arr(out, "RESOURCES", state.resources[:f].reshape(-1))
    _arr(out, "KILL_XP", state.kill_xp[:f])
    _arr(out, "VICTORY_POINTS", state.victory_points[:f])
    _arr(out, "ALIVE", state.alive[:f].astype(np.int32))
    out.write("END\n")
