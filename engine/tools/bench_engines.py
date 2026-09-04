"""
Per-agent timing for both engines on identical games.

    python engine/tools/bench_engines.py old
    python engine/tools/bench_engines.py new

Run each and compare the tables by hand. Separate processes, because both engines
want the `engine` module name.

These numbers are NOT comparable to PLAN.md §0's original baselines: the §3.2
sorted() change altered the seed -> map mapping, so the games themselves differ.
Both engines here play the SAME games as each other, which is what matters.
"""

import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(_HERE))

CASES = [
    ("random", 7, 8), ("greedy", 7, 8), ("heuristic", 7, 8), ("vanguard", 7, 8),
    ("marshal", 7, 8), ("tactician", 7, 8), ("tactician", 5, 6),
]
GAMES = 5
MAX_TURNS = 60


def main():
    which = sys.argv[1]
    sys.path.insert(0, _REPO)
    if which == "old":
        sys.path.insert(0, _HERE)
        import _bootstrap
        _bootstrap.install()
    else:
        import engine  # noqa: F401
    import tournament

    print(f"{'agent':<12} {'board':<8} {'games':>5} {'total s':>9} {'s/game':>9} {'turns':>7}")
    for agent, radius, nf in CASES:
        t0 = time.perf_counter()
        turns = 0
        for g in range(GAMES):
            r = tournament.play_game({f: agent for f in range(nf)}, radius, nf, 7000 + g,
                                     max_turns=MAX_TURNS)
            turns += r["turns"]
        dt = time.perf_counter() - t0
        print(f"{agent:<12} r{radius}f{nf:<5} {GAMES:>5} {dt:>9.3f} {dt / GAMES:>9.4f} "
              f"{turns / GAMES:>7.1f}")


if __name__ == "__main__":
    main()
