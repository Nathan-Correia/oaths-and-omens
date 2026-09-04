"""
M6c gate: the three replay files, written natively, must be byte-identical to
Python's.

Byte-identical rather than semantically-equal on purpose. The files exist to be
read by web_visualizer.html after every trace of Python is gone (PLAN.md §1.3),
so "close enough" is not a standard that can be checked later - and comparing
bytes catches key-order and separator drift that a dict comparison would silently
accept.

Runs run.py's pipeline in-process against engine_old for a given seed, writes the
files to one directory, runs oo_run for the same seed into another, and diffs.

Usage:
    python engine/tools/compare_replay_json.py <oo_run.exe> [seeds...]
"""

import filecmp
import json
import os
import random
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _bootstrap  # noqa: E402

_bootstrap.install()

from engine.placement import run_city_setup  # noqa: E402
from engine.setup import create_initial_state  # noqa: E402
from engine.state import TERRAIN_TYPES  # noqa: E402
from engine.turn import check_game_end, run_turn_and_log  # noqa: E402
from agents import compose_agents  # noqa: E402
from agents.tactician_agent import make_tactician_agents  # noqa: E402
from agents.greedy_agent import make_greedy_agents  # noqa: E402
from agents.marshal_agent import make_marshal_agents  # noqa: E402

BUILDERS = {
    "tactician": make_tactician_agents,
    "greedy": make_greedy_agents,
    "marshal": make_marshal_agents,
}

FILES = ["board_state.json", "terrain_gen_log.json", "city_placement_log.json"]


def run_python(out_dir, radius, num_factions, seed, max_turns, agent):
    """Exactly run.py's pipeline, writing into out_dir."""
    rng = random.Random(seed)
    terrain_log = []
    state = create_initial_state(radius=radius, num_factions=num_factions, seed=seed,
                                 terrain_log=terrain_log)

    with open(os.path.join(out_dir, "terrain_gen_log.json"), "w") as f:
        json.dump({"radius": radius, "steps": terrain_log}, f)

    terrain_map = {
        f"{q}_{r}_{s}": TERRAIN_TYPES[int(t)]
        for (q, r, s), t in zip(state.grid.coords, state.terrain)
    }

    decide = compose_agents({f: agent for f in range(num_factions)},
                            {agent: lambda: BUILDERS[agent](num_factions, seed=seed)})
    placement_log = []
    state = run_city_setup(state, decide[6], decide[7], decide[8], rng, log=placement_log)

    with open(os.path.join(out_dir, "city_placement_log.json"), "w") as f:
        json.dump({"radius": radius, "num_factions": num_factions,
                   "terrain": terrain_map, "steps": placement_log}, f)

    turns = []
    while not check_game_end(state, max_turns=max_turns):
        state, record = run_turn_and_log(state, decide[0], decide[1], decide[2], decide[3],
                                         decide[4], decide[5], rng=rng)
        turns.append(record)

    with open(os.path.join(out_dir, "board_state.json"), "w") as f:
        json.dump({"radius": radius, "num_factions": num_factions,
                   "terrain": terrain_map, "turns": turns}, f)
    return len(turns)


def first_difference(a_path, b_path):
    a = open(a_path, "rb").read()
    b = open(b_path, "rb").read()
    if a == b:
        return None
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    lo = max(0, i - 90)
    return (f"    first differing byte at offset {i} (python {len(a)} bytes, native {len(b)})\n"
            f"    python: ...{a[lo:i + 90].decode('utf-8', 'replace')}\n"
            f"    native: ...{b[lo:i + 90].decode('utf-8', 'replace')}")


def coverage(path, acc):
    """What rare JSON branches the compared corpus actually exercised.

    A byte-identical result over a corpus that never produces a "reason": "cap"
    dismount, a null winner, or a temple upgrade has not tested those code paths
    at all - the same trap as PLAN.md §3.3a. So the gate reports its own coverage
    rather than leaving it assumed.
    """
    d = json.load(open(path))
    if "turns" not in d:
        for st in d.get("steps", []):
            if "type" in st:
                acc["placement_" + st["type"]] += 1
        return
    for t in d["turns"]:
        for c in t["checkpoints"]:
            for h in c:
                if h["city"]:
                    acc["upgrade_" + str(h["city"]["upgrade"])] += 1
                if h["troops"] and h["troops"]["frozen"]:
                    acc["frozen_troops"] += 1
                if h["battle"]:
                    acc["battle_snapshot"] += 1
        for e in t["battle_events"]:
            acc["battle_event"] += 1
            if e["winner"] is None:
                acc["winner_null"] += 1
            acc["rectification"] += len(e["rectification"])
            acc["structure_kills"] += len(e["structure_phase"])
            acc["archer_kills"] += len(e["archer_phase"])
            for r in e["rounds"]:
                acc["round"] += 1
                for x in r["deaths"]:
                    acc["death"] += 1
                for x in r["dismounts"]:
                    acc["dismount_cap" if x.get("reason") == "cap" else "dismount"] += 1
                for v in r["target_choices_submitted"].values():
                    if v is None:
                        acc["target_null"] += 1


def main():
    exe = sys.argv[1]
    seeds = [int(x) for x in sys.argv[2:]] or [11, 12, 13, 21, 22, 23, 31, 32]
    # Longer games and a spread of agents, chosen so the corpus reaches the rare
    # branches the coverage report below checks for.
    cases = [(7, 8, "tactician", 40), (5, 6, "greedy", 60), (4, 4, "marshal", 60),
             (8, 8, "greedy", 60), (7, 8, "greedy", 60)]

    failures = 0
    checked = 0
    acc = __import__("collections").Counter()
    for radius, num_factions, agent, max_turns in cases:
        for seed in seeds:
            with tempfile.TemporaryDirectory() as py_dir, tempfile.TemporaryDirectory() as cpp_dir:
                turns = run_python(py_dir, radius, num_factions, seed, max_turns, agent)
                subprocess.run(
                    [exe, "--radius", str(radius), "--factions", str(num_factions),
                     "--seed", str(seed), "--max-turns", str(max_turns), "--agent", agent,
                     "--out-dir", cpp_dir],
                    check=True, stdout=subprocess.DEVNULL)

                tag = f"{agent}-r{radius}f{num_factions}s{seed} ({turns} turns)"
                for name in FILES:
                    checked += 1
                    a = os.path.join(py_dir, name)
                    b = os.path.join(cpp_dir, name)
                    coverage(a, acc)
                    if filecmp.cmp(a, b, shallow=False):
                        continue
                    failures += 1
                    print(f"DIFFER {tag} {name}")
                    if failures <= 3:
                        print(first_difference(a, b))

    print(f"{checked - failures}/{checked} files byte-identical")
    print("")
    print("coverage of the compared corpus:")
    for k in sorted(acc):
        print(f"  {k:<22} {acc[k]}")
    missing = [k for k in ("upgrade_barracks", "upgrade_workshop", "upgrade_temple",
                           "dismount_cap", "winner_null", "frozen_troops", "battle_snapshot",
                           "rectification", "structure_kills", "archer_kills", "target_null",
                           "placement_swap", "placement_keep", "placement_draft_auto")
               if acc[k] == 0]
    if missing:
        print("")
        print("WARNING - never exercised: " + ", ".join(missing))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
