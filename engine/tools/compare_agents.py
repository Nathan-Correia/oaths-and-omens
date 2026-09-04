"""
Decision-level parity between the Python agents and their native ports (M6a).

The game is driven by the PYTHON agents, so it follows exactly the reference
trajectory. At every decision point the NATIVE agent is asked about the same
state and its answer is compared. That pinpoints the first differing decision
with full context, instead of leaving you to infer a cause from a diverging final
score - the same "compare the decision, not the outcome" lesson as PLAN.md §3.3c.

Both agent sets are constructed with the same seed, so their per-faction RNG
streams start aligned. They stay aligned only while their decisions agree and
consume the same number of draws, which is exactly what is being tested.

Usage:
    python engine/tools/compare_agents.py [agent ...]
"""

import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _REPO)

import engine  # noqa: E402  (the shim package -> compiled module)
from oo_engine import NativeAgentSet  # noqa: E402

from engine.placement import run_city_setup  # noqa: E402
from engine.setup import create_initial_state  # noqa: E402
from engine.turn import run_turn  # noqa: E402
from agents import compose_agents  # noqa: E402
from agents.denier_agent import make_denier_agents  # noqa: E402
from agents.greedy_agent import make_greedy_agents  # noqa: E402
from agents.heuristic_agent import make_heuristic_agents  # noqa: E402
from agents.hussar_agent import make_hussar_agents  # noqa: E402
from agents.legion_agent import make_legion_agents  # noqa: E402
from agents.marshal_agent import make_marshal_agents  # noqa: E402
from agents.random_agent import make_random_agents  # noqa: E402
from agents.sentinel_agent import make_sentinel_agents  # noqa: E402
from agents.tactician_agent import make_tactician_agents  # noqa: E402
from agents.turtle_agent import make_turtle_agents  # noqa: E402
from agents.vanguard_agent import make_vanguard_agents  # noqa: E402
from agents.warlord_agent import make_warlord_agents  # noqa: E402

BUILDERS = {
    "random": make_random_agents,
    "greedy": make_greedy_agents,
    "heuristic": make_heuristic_agents,
    "vanguard": make_vanguard_agents,
    "marshal": make_marshal_agents,
    "turtle": make_turtle_agents,
    "denier": make_denier_agents,
    "warlord": make_warlord_agents,
    "legion": make_legion_agents,
    "hussar": make_hussar_agents,
    "sentinel": make_sentinel_agents,
    "tactician": make_tactician_agents,
}

SIZES = [(7, 8), (5, 6), (4, 4)]
SEEDS = [5000, 5001, 5002]
TURNS = 12

# Overridable so a specific divergence can be chased further into a game than
# the default sweep goes.
import os as _os
if _os.environ.get("OO_CMP_SIZES"):
    SIZES = [tuple(int(x) for x in p.split(",")) for p in _os.environ["OO_CMP_SIZES"].split(";")]
if _os.environ.get("OO_CMP_SEEDS"):
    SEEDS = [int(x) for x in _os.environ["OO_CMP_SEEDS"].split(",")]
if _os.environ.get("OO_CMP_TURNS"):
    TURNS = int(_os.environ["OO_CMP_TURNS"])


def norm_buy(actions):
    """Canonical form for comparison - the dicts differ only in key order."""
    return [tuple(sorted(a.items())) for a in (actions or [])]


def norm_rect(entries):
    return [(e["origin_hex"], tuple(int(u) for u in e["units"])) for e in (entries or [])]


def compare(agent_key, radius, num_factions, seed, report):
    py_decide = compose_agents(
        {f: agent_key for f in range(num_factions)},
        {agent_key: lambda: BUILDERS[agent_key](num_factions, seed=seed)})
    native = NativeAgentSet(agent_key, num_factions, seed)

    tag = f"{agent_key}-r{radius}f{num_factions}s{seed}"
    stats = {"checked": 0, "diff": 0}

    def check(kind, faction, got_py, got_native, extra=""):
        stats["checked"] += 1
        if got_py == got_native:
            return
        stats["diff"] += 1
        if stats["diff"] <= 3:
            report.append(f"{tag} {kind} f{faction} {extra}\n"
                          f"    python: {got_py}\n"
                          f"    native: {got_native}")

    d_buy, d_move, d_cav, d_target, d_rect, d_res, d_place, d_draft, d_swap = py_decide
    turn_no = [0]

    def w_buy(state, faction, legal):
        got = d_buy[faction](state, faction, legal)
        check(f"buy(turn {turn_no[0]})", faction, norm_buy(got),
              norm_buy(native.decide_buy(state, faction)))
        return got

    def w_move(state, faction, step, mask):
        got = d_move[faction](state, faction, step, mask)
        check(f"move(turn {turn_no[0]},step {step})", faction, got,
              native.decide_movement(state, faction, step))
        return got

    def w_cav(state, faction, step, mask):
        got = d_cav[faction](state, faction, step, mask)
        check(f"cav(turn {turn_no[0]},step {step})", faction, got,
              native.decide_cavalry(state, faction, step))
        return got

    def w_target(state, hex_index, faction):
        got = d_target[faction](state, hex_index, faction)
        check(f"target(turn {turn_no[0]})", faction, got,
              native.decide_target(state, hex_index, faction), f"hex={hex_index}")
        return got

    def w_rect(state, hex_index, winner, cap):
        got = d_rect[winner](state, hex_index, winner, cap)
        check(f"rect(turn {turn_no[0]})", winner, norm_rect(got),
              norm_rect(native.decide_rectification(state, hex_index, winner, cap)),
              f"hex={hex_index} cap={cap}")
        return got

    def w_res(state, faction, hex_index):
        got = d_res[faction](state, faction, hex_index)
        check(f"resource(turn {turn_no[0]})", faction, got,
              native.decide_resource_choice(state, faction, hex_index), f"hex={hex_index}")
        return got

    def w_place(state, faction, mask):
        got = d_place[faction](state, faction, mask)
        check("placement", faction, int(got), int(native.decide_placement(state, faction)))
        return got

    def w_draft(state, faction, pool):
        got = d_draft[faction](state, faction, pool)
        check("draft", faction, int(got), int(native.decide_draft(state, faction, list(pool))))
        return got

    def w_swap(state, faction, leftover, placer, placer_hex):
        got = d_swap[faction](state, faction, leftover, placer, placer_hex)
        check("swap", faction, bool(got),
              bool(native.decide_swap(state, faction, leftover, placer, placer_hex)))
        return got

    fs = range(num_factions)
    wrap = [{f: w for f in fs} for w in
            (w_buy, w_move, w_cav, w_target, w_rect, w_res, w_place, w_draft, w_swap)]

    rng = random.Random(seed)
    state = create_initial_state(radius=radius, num_factions=num_factions, seed=seed)
    state = run_city_setup(state, wrap[6], wrap[7], wrap[8], rng)
    for t in range(TURNS):
        turn_no[0] = t
        state = run_turn(state, wrap[0], wrap[1], wrap[2], wrap[3], wrap[4], wrap[5], rng=rng)
    return stats


def main():
    wanted = sys.argv[1:] or list(BUILDERS)
    report = []
    total = {"checked": 0, "diff": 0}
    print(f"{'agent':<12} {'decisions':>10} {'differing':>10}")
    for agent_key in wanted:
        checked = diff = 0
        for radius, nf in SIZES:
            for seed in SEEDS:
                s = compare(agent_key, radius, nf, seed, report)
                checked += s["checked"]
                diff += s["diff"]
        total["checked"] += checked
        total["diff"] += diff
        print(f"{agent_key:<12} {checked:>10} {diff:>10}")
    print(f"{'TOTAL':<12} {total['checked']:>10} {total['diff']:>10}")
    if report:
        print("\nfirst differences:")
        for line in report[:12]:
            print(" ", line)
    return 1 if total["diff"] else 0


if __name__ == "__main__":
    sys.exit(main())
