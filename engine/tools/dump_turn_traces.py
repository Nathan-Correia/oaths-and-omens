"""
Dumps full-turn decision traces from engine_old for the C++ engine to replay.

THE IDEA (PLAN.md §3.3): rather than reimplement each agent's policy in C++ just to
compare turns, record what the Python agents actually DECIDED, in order, and have
the C++ engine replay those decisions. No policy is duplicated, and any agent can
be used as a trace source - including tactician, whose search would be miserable to
mirror by hand.

It also checks more than the resulting state. The replaying side asserts that the
C++ engine asks for decisions in the same ORDER and with the same ARGUMENTS as
Python did. A C++ battle that runs an extra round, or asks the wrong faction for a
target, is caught at the point of divergence rather than as a mystery diff several
phases later.

Each turn gets a FRESH rng seeded from (game seed, turn number) rather than
threading one generator through the game. That makes every case independently
replayable, at the cost of the traced game's dice differing from a normal run -
which does not matter here, since what is under test is the engine, not any
particular game.

Usage:  python engine/tools/dump_turn_traces.py > engine/tests/data/turn_traces.txt
"""

import copy
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _bootstrap  # noqa: E402

_bootstrap.install()

from engine.placement import run_city_setup  # noqa: E402
from engine.setup import create_initial_state  # noqa: E402
from engine.turn import run_turn  # noqa: E402
from agents import compose_agents  # noqa: E402
from agents.greedy_agent import make_greedy_agents  # noqa: E402
from agents.heuristic_agent import make_heuristic_agents  # noqa: E402
from agents.marshal_agent import make_marshal_agents  # noqa: E402
from agents.random_agent import make_random_agents  # noqa: E402
from agents.tactician_agent import make_tactician_agents  # noqa: E402
from agents.vanguard_agent import make_vanguard_agents  # noqa: E402

from state_io import write_state  # noqa: E402

BUILDERS = {
    "random": make_random_agents,
    "greedy": make_greedy_agents,
    "heuristic": make_heuristic_agents,
    "vanguard": make_vanguard_agents,
    "marshal": make_marshal_agents,
    "tactician": make_tactician_agents,
}

UNIT_INDEX = {"infantry": 0, "cavalry": 1, "archers": 2}
UPGRADE_INDEX = {"barracks": 0, "workshop": 1, "temple": 2}
BUY_TYPE_INDEX = {
    "buy_infantry": 0,
    "convert_to_special": 1,
    "build_outpost": 2,
    "upgrade_outpost": 3,
}


def _clone(state):
    new = copy.copy(state)
    for field in ("terrain", "city_owner", "is_capital", "outpost_upgrade", "city_placer",
                  "capital_settle_order", "army_faction", "army_units", "frozen", "locked",
                  "battle_faction", "battle_origin", "battle_units", "battle_moved",
                  "battle_round", "gold", "resources", "kill_xp", "victory_points", "alive"):
        setattr(new, field, getattr(state, field).copy())
    new.battle_order = list(state.battle_order)
    return new


def _buy_action_tokens(action):
    """<type> <hex> <unit_type> <upgrade>, with unused fields zeroed."""
    kind = BUY_TYPE_INDEX[action["type"]]
    if action["type"] == "buy_infantry":
        return f"{kind} {action['city_hex']} 0 0"
    if action["type"] == "convert_to_special":
        return f"{kind} {action['hex']} {UNIT_INDEX[action['unit_type']]} 0"
    if action["type"] == "build_outpost":
        return f"{kind} {action['hex']} {UNIT_INDEX[action['unit_type']]} 0"
    return f"{kind} {action['hex']} 0 {UPGRADE_INDEX[action['upgrade']]}"


class Recorder:
    """Wraps a set of agent callbacks, logging every decision in call order."""

    def __init__(self, decide, num_factions):
        self.lines = []
        (d_buy, d_move, d_cav, d_target, d_rect, d_res, _p, _d, _s) = decide
        self.num_factions = num_factions

        def buy(state, faction, legal, _inner=d_buy):
            chosen = _inner[faction](state, faction, legal) or []
            tokens = " ".join(_buy_action_tokens(a) for a in chosen)
            self.lines.append(f"B {faction} {len(chosen)}" + (" " + tokens if tokens else ""))
            return chosen

        def move(state, faction, step, legal_mask, _inner=d_move):
            got = _inner[faction](state, faction, step, legal_mask)
            h, d = (-1, -1) if got is None else (int(got[0]), int(got[1]))
            self.lines.append(f"M {faction} {step} {h} {d}")
            return got

        def cav(state, faction, step, legal_mask, _inner=d_cav):
            got = _inner[faction](state, faction, step, legal_mask)
            h, d = (-1, -1) if got is None else (int(got[0]), int(got[1]))
            self.lines.append(f"C {faction} {step} {h} {d}")
            return got

        def target(state, hex_index, faction, _inner=d_target):
            got = _inner[faction](state, hex_index, faction)
            self.lines.append(f"T {hex_index} {faction} {-1 if got is None else int(got)}")
            return got

        def rect(state, hex_index, winner, cap, _inner=d_rect):
            got = _inner[winner](state, hex_index, winner, cap) or []
            parts = [f"R {hex_index} {winner} {cap} {len(got)}"]
            for entry in got:
                origin = entry["origin_hex"]
                origin = -1 if origin is None else int(origin)
                u = entry["units"]
                parts.append(f"{origin} {int(u[0])} {int(u[1])} {int(u[2])}")
            self.lines.append(" ".join(parts))
            return got

        def res(state, faction, hex_index, _inner=d_res):
            got = _inner[faction](state, faction, hex_index)
            self.lines.append(f"P {faction} {hex_index} {1 if got == 'iron' else 0}")
            return got

        factions = range(num_factions)
        self.decide_buy = {f: buy for f in factions}
        self.decide_movement = {f: move for f in factions}
        self.decide_cavalry = {f: cav for f in factions}
        self.decide_target = {f: target for f in factions}
        self.decide_rectification = {f: rect for f in factions}
        self.decide_resource_choice = {f: res for f in factions}


# radius, factions, seed, turns, agent. A spread of board sizes, player counts and
# playstyles: random for raw code-path coverage (it fuzzes everything), greedy and
# marshal for realistic play that actually builds and fights, tactician because its
# search produces move choices no hand-written policy would.
SCENARIOS = [
    (4, 4, 101, 14, "random"),
    (5, 6, 102, 14, "random"),
    (7, 8, 103, 12, "random"),
    (8, 8, 104, 10, "random"),
    (5, 10, 105, 10, "random"),
    (7, 8, 201, 16, "greedy"),
    (5, 6, 202, 16, "greedy"),
    (4, 4, 203, 14, "greedy"),
    (8, 8, 204, 12, "greedy"),
    (7, 8, 301, 14, "heuristic"),
    (7, 8, 302, 14, "vanguard"),
    (7, 8, 401, 14, "marshal"),
    (5, 6, 402, 12, "marshal"),
    (7, 8, 501, 8, "tactician"),
]


def main():
    out = sys.stdout
    cases = []

    for radius, num_factions, seed, turns, agent_key in SCENARIOS:
        setup_rng = random.Random(seed)
        state = create_initial_state(radius=radius, num_factions=num_factions, seed=seed)
        decide = compose_agents({f: agent_key for f in range(num_factions)},
                                {agent_key: lambda: BUILDERS[agent_key](num_factions, seed=seed)})
        state = run_city_setup(state, decide[6], decide[7], decide[8], setup_rng)

        rec = Recorder(decide, num_factions)
        for turn in range(turns):
            before = _clone(state)
            rec.lines = []
            turn_seed = seed * 7919 + turn
            run_turn(state, rec.decide_buy, rec.decide_movement, rec.decide_cavalry,
                     rec.decide_target, rec.decide_rectification, rec.decide_resource_choice,
                     rng=random.Random(turn_seed))
            cases.append((f"{agent_key}-r{radius}f{num_factions}s{seed}t{turn}", turn_seed, before,
                          list(rec.lines), _clone(state)))

    out.write(f"TURN_CASES {len(cases)}\n")
    for name, turn_seed, before, lines, after in cases:
        out.write(f"TURN_CASE {name}\n")
        out.write(f"SEED {turn_seed}\n")
        write_state(out, before)
        out.write(f"DECISIONS {len(lines)}\n")
        for line in lines:
            out.write(line + "\n")
        write_state(out, after)


if __name__ == "__main__":
    main()
