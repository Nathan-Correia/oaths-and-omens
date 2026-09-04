"""
Terrain-generation and capital-setup parity cases.

TWO SECTIONS, because the two halves of setup have very different shapes:

  TERRAIN cases are PURE - a radius and a seed in, a map out, no decisions
  anywhere. So there is nothing to record: the C++ engine regenerates from the
  same seed and the maps must be identical, hex for hex, plus the generation log
  in the same order. This is the M4 gate, and the thing §3.2's sorted() change was
  made to enable: full-pipeline parity from a seed alone.

  SETUP cases involve agents (placement, draft, swap), so they use the same
  decision-trace replay as the turn traces - recording the agent's RAW return
  value, before engine_old validates it. That matters: an invalid answer falls
  back to a random legal choice, which CONSUMES RNG, so replaying the raw value
  reproduces the fallback path too.

Usage:  python engine/tools/dump_setup_cases.py > engine/tests/data/setup_cases.txt
"""

import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _bootstrap  # noqa: E402

_bootstrap.install()

from engine.geometry import HexGrid  # noqa: E402
from engine.placement import run_city_setup  # noqa: E402
from engine.setup import create_initial_state, generate_terrain  # noqa: E402
from engine.state import TERRAIN_TO_INDEX  # noqa: E402
from agents import compose_agents  # noqa: E402
from agents.greedy_agent import make_greedy_agents  # noqa: E402
from agents.heuristic_agent import make_heuristic_agents  # noqa: E402
from agents.marshal_agent import make_marshal_agents  # noqa: E402
from agents.random_agent import make_random_agents  # noqa: E402
from agents.tactician_agent import make_tactician_agents  # noqa: E402

from state_io import write_state  # noqa: E402

BUILDERS = {
    "random": make_random_agents,
    "greedy": make_greedy_agents,
    "heuristic": make_heuristic_agents,
    "marshal": make_marshal_agents,
    "tactician": make_tactician_agents,
}

# Radius 8 is the largest the generator supports - BAG_COUNTS totals 250 hexes,
# short of a radius-9 board's 271, so generation crashes past 8 (PLAN.md §9).
TERRAIN_RADII = [1, 2, 3, 4, 5, 6, 7, 8]
TERRAIN_SEEDS_PER_RADIUS = 40

SETUP_SCENARIOS = [
    # radius, factions, seed, agent. Faction counts chosen to straddle the 5-7
    # edge-ban window on both sides, since that changes which mask tier applies.
    (4, 2, 900, "random"),
    (4, 4, 901, "random"),
    (5, 5, 902, "random"),
    (5, 6, 903, "random"),
    (7, 7, 904, "random"),
    (7, 8, 905, "random"),
    (7, 10, 906, "random"),
    (8, 8, 907, "random"),
    (4, 4, 910, "greedy"),
    (5, 6, 911, "greedy"),
    (7, 8, 912, "greedy"),
    (8, 10, 913, "greedy"),
    (7, 8, 920, "heuristic"),
    (7, 8, 921, "marshal"),
    (7, 8, 922, "tactician"),
    (5, 6, 923, "tactician"),
    # A tiny board with many factions, to push the mask into its relaxation tiers:
    # radius 3 is 37 hexes, and 8 capitals at distance >= 3 will not fit.
    (3, 8, 930, "random"),
    (3, 10, 931, "greedy"),
    (2, 6, 932, "random"),
]

KIND_INDEX = {"place": 0, "draft": 1, "draft_auto": 2, "keep": 3, "swap": 4}


def main():
    out = sys.stdout

    # --- terrain ------------------------------------------------------------
    terrain_cases = []
    for radius in TERRAIN_RADII:
        grid = HexGrid(radius)
        for s in range(TERRAIN_SEEDS_PER_RADIUS):
            seed = radius * 100003 + s
            log = []
            terrain = generate_terrain(grid, random.Random(seed), log=log)
            terrain_cases.append((radius, seed, terrain, log))

    out.write(f"TERRAIN_CASES {len(terrain_cases)}\n")
    for radius, seed, terrain, log in terrain_cases:
        out.write(f"TERRAIN_CASE {radius} {seed}\n")
        out.write(f"TERRAIN {len(terrain)} " + " ".join(str(int(t)) for t in terrain) + "\n")
        out.write(f"LOG {len(log)}\n")
        for e in log:
            out.write(f"{e['q']} {e['r']} {e['s']} {TERRAIN_TO_INDEX[e['terrain']]} {e['round']}\n")

    # --- capital setup ------------------------------------------------------
    setup_cases = []
    for radius, num_factions, seed, agent_key in SETUP_SCENARIOS:
        state = create_initial_state(radius=radius, num_factions=num_factions, seed=seed)
        decide = compose_agents({f: agent_key for f in range(num_factions)},
                                {agent_key: lambda: BUILDERS[agent_key](num_factions, seed=seed)})
        d_place, d_draft, d_swap = decide[6], decide[7], decide[8]

        lines = []

        def rec_place(st, faction, legal_mask, _inner=d_place):
            got = _inner[faction](st, faction, legal_mask)
            # The MASK is recorded, not just the choice. Mutation testing showed
            # that comparing only the choice leaves legal_placement_mask itself
            # untested: a wrong mask that is merely LARGER still validates the
            # recorded pick, so the edge ban could be dropped entirely with no
            # test failing. Same class of gap as get_legal_buy_actions in M3.
            legal = [i for i, ok in enumerate(legal_mask) if ok]
            lines.append(f"L {faction} {-1 if got is None else int(got)} {len(legal)}"
                         + ("".join(f" {i}" for i in legal) if legal else ""))
            return got

        def rec_draft(st, faction, legal_pool, _inner=d_draft):
            got = _inner[faction](st, faction, legal_pool)
            pool = [int(h) for h in legal_pool]
            lines.append(f"D {faction} {-1 if got is None else int(got)} {len(pool)}"
                         + ("".join(f" {h}" for h in pool) if pool else ""))
            return got

        def rec_swap(st, faction, leftover, placer, placer_hex, _inner=d_swap):
            got = _inner[faction](st, faction, leftover, placer, placer_hex)
            lines.append(f"W {faction} {1 if got else 0} 0")
            return got

        factions = range(num_factions)
        wrapped = ({f: rec_place for f in factions}, {f: rec_draft for f in factions},
                   {f: rec_swap for f in factions})

        before = state
        import copy
        after = copy.copy(state)
        for field in ("terrain", "city_owner", "is_capital", "outpost_upgrade", "city_placer",
                      "capital_settle_order", "army_faction", "army_units", "frozen", "locked",
                      "battle_faction", "battle_origin", "battle_units", "battle_moved",
                      "battle_round", "gold", "resources", "kill_xp", "victory_points", "alive"):
            setattr(after, field, getattr(state, field).copy())
        after.battle_order = list(state.battle_order)

        setup_seed = seed * 31 + 7
        log = []
        run_city_setup(after, wrapped[0], wrapped[1], wrapped[2], random.Random(setup_seed),
                       log=log)
        setup_cases.append((f"{agent_key}-r{radius}f{num_factions}s{seed}", setup_seed, before,
                            lines, after, log, radius, num_factions, seed))

    out.write(f"SETUP_CASES {len(setup_cases)}\n")
    for name, setup_seed, before, lines, after, log, radius, nf, game_seed in setup_cases:
        out.write(f"SETUP_CASE {name}\n")
        # PARAMS lets the C++ side BUILD the before-state from the seed alone via
        # create_initial_state rather than reading it in, so each case is a real
        # from-seed pipeline check (terrain generation + starting gold/kill-XP)
        # composed with the placement replay, not just the placement half.
        out.write(f"PARAMS {radius} {nf} {game_seed}\n")
        out.write(f"SEED {setup_seed}\n")
        write_state(out, before)
        out.write(f"DECISIONS {len(lines)}\n")
        for line in lines:
            out.write(line + "\n")
        write_state(out, after)
        out.write(f"LOG {len(log)}\n")
        for e in log:
            kind = KIND_INDEX[e["type"]]
            pf = e.get("placer_faction", -1)
            pq = e.get("placer_q", 0)
            pr = e.get("placer_r", 0)
            ps = e.get("placer_s", 0)
            out.write(f"{kind} {e['faction']} {e['q']} {e['r']} {e['s']} {pf} {pq} {pr} {ps}\n")


if __name__ == "__main__":
    main()
