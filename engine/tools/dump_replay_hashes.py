"""
Golden hashes of the three replay files, as produced by the Python pipeline.

tests/test_replay.cpp regenerates each file natively and hashes the bytes. Hashes
rather than the files themselves because a single board_state.json is ~270 KB and
several would dominate the repo - but the check is still byte-exact, which is the
point (PLAN.md §1.3).

Checked in, so ctest needs no Python. Regenerate when the format or the reference
changes.

Usage:  python engine/tools/dump_replay_hashes.py > engine/tests/data/replay_hashes.txt
"""

import hashlib
import json
import os
import random
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _bootstrap  # noqa: E402

_bootstrap.install()

from compare_replay_json import FILES, run_python  # noqa: E402

# Kept small and varied: enough games to reach battles, upgrades and swaps
# without making regeneration slow.
CASES = [
    (4, 4, "greedy", 20, 7),
    (5, 6, "marshal", 20, 11),
    (7, 8, "greedy", 15, 13),
    (7, 8, "tactician", 10, 21),
]


def main():
    rows = []
    for radius, num_factions, agent, max_turns, seed in CASES:
        with tempfile.TemporaryDirectory() as d:
            run_python(d, radius, num_factions, seed, max_turns, agent)
            for name in FILES:
                data = open(os.path.join(d, name), "rb").read()
                digest = hashlib.sha256(data).hexdigest()
                rows.append(f"{agent} {radius} {num_factions} {seed} {max_turns} {name} "
                            f"{len(data)} {digest}")

    sys.stdout.write(f"REPLAY_HASHES {len(rows)}\n")
    for row in rows:
        sys.stdout.write(row + "\n")


if __name__ == "__main__":
    main()
