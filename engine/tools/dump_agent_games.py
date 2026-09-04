"""
Golden results for whole games played by the reference Python agents.

tests/test_agents.cpp replays the same matrix with the NATIVE agents and must get
identical outcomes. Checked in, so `ctest` needs no Python (PLAN.md §1.2) - Python
is only needed to regenerate this when an agent or the matrix changes.

The matrix here MUST match apps/oo_tournament.cpp's kMatrix and
tests/test_agents.cpp, entry for entry and in order.

Usage:  python engine/tools/dump_agent_games.py > engine/tests/data/agent_games.txt
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _bootstrap  # noqa: E402

_bootstrap.install()

import tournament  # noqa: E402

MATRIX = [
    ("random", 7, 8), ("random", 4, 4), ("random", 5, 6), ("random", 8, 8),
    ("greedy", 7, 8), ("greedy", 4, 4), ("greedy", 5, 6), ("greedy", 8, 8),
    ("heuristic", 7, 8), ("heuristic", 5, 6), ("heuristic", 4, 4),
    ("vanguard", 7, 8), ("vanguard", 5, 6),
    ("marshal", 7, 8), ("marshal", 5, 6), ("marshal", 8, 8),
]
SEEDS_PER_ENTRY = 5
BASE_SEED = 5000
MAX_TURNS = 60


def main():
    rows = []
    for agent, radius, num_factions in MATRIX:
        for s in range(SEEDS_PER_ENTRY):
            seed = BASE_SEED + s
            r = tournament.play_game({f: agent for f in range(num_factions)}, radius,
                                     num_factions, seed, max_turns=MAX_TURNS)
            winner = -1 if r["winner"] is None else r["winner"]
            vp = " ".join(str(r["vp"][f]) for f in range(num_factions))
            rows.append(f"{agent} {radius} {num_factions} {seed} {winner} {r['turns']} {vp}")

    sys.stdout.write(f"AGENT_GAMES {len(rows)}\n")
    for row in rows:
        sys.stdout.write(row + "\n")


if __name__ == "__main__":
    main()
