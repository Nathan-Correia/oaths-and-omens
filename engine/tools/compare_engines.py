"""
M5 acceptance gate: run the SAME games through both engines and compare results.

Usage:
    python engine/tools/compare_engines.py old  > old.json
    python engine/tools/compare_engines.py new  > new.json
    python engine/tools/compare_engines.py diff old.json new.json

Two processes, because both engines want to be importable as `engine` and only
one can win that name per interpreter.

What is compared is the full outcome of each game - winner, turn count, and every
faction's victory points - for identical (agents, radius, factions, seed) inputs.
Anything less would let a divergence hide: a game can end on the same turn with a
different board.

NOTE: these results are NOT comparable to the pre-M4 baselines in PLAN.md §0. The
sorted() change in engine_old/setup.py (§3.2) altered the seed -> map mapping, so
a given seed now produces a different (equally valid) board. Both engines here use
the fixed generator, so they must agree with each other.
"""

import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(_HERE))

# agents x board sizes. Deliberately includes tactician, whose search calls the
# engine thousands of extra times per turn through the binding layer.
# The M6a subset: the agents implemented natively so far. MUST match
# apps/oo_tournament.cpp's kMatrix entry for entry, in order.
MATRIX_M6A = [
    ("random", 7, 8), ("random", 4, 4), ("random", 5, 6), ("random", 8, 8),
    ("greedy", 7, 8), ("greedy", 4, 4), ("greedy", 5, 6), ("greedy", 8, 8),
    ("heuristic", 7, 8), ("vanguard", 7, 8), ("marshal", 7, 8), ("marshal", 5, 6),
]

MATRIX = [
    ("random", 7, 8), ("random", 4, 4), ("random", 5, 6), ("random", 8, 8),
    ("greedy", 7, 8), ("greedy", 4, 4), ("greedy", 5, 6), ("greedy", 8, 8),
    ("heuristic", 7, 8), ("turtle", 7, 8), ("denier", 7, 8), ("vanguard", 7, 8),
    ("warlord", 7, 8), ("legion", 7, 8), ("hussar", 7, 8), ("sentinel", 7, 8),
    ("marshal", 7, 8), ("marshal", 5, 6), ("tactician", 5, 6), ("tactician", 7, 8),
]
SEEDS_PER_ENTRY = 5
MAX_TURNS = 60


def run(which, matrix=None):
    matrix = matrix or MATRIX
    sys.path.insert(0, _REPO)
    if which == "old":
        sys.path.insert(0, _HERE)
        import _bootstrap
        _bootstrap.install()
    else:
        # The shim package at engine/ re-exports the compiled module.
        import engine  # noqa: F401

    import tournament

    results = []
    for agent, radius, num_factions in matrix:
        for s in range(SEEDS_PER_ENTRY):
            seed = 5000 + s
            r = tournament.play_game({f: agent for f in range(num_factions)}, radius,
                                     num_factions, seed, max_turns=MAX_TURNS)
            results.append({
                "agent": agent, "radius": radius, "factions": num_factions, "seed": seed,
                "winner": r["winner"], "turns": r["turns"],
                "vp": {str(k): v for k, v in r["vp"].items()},
            })
    json.dump(results, sys.stdout)


def diff(path_a, path_b):
    a = json.load(open(path_a))
    b = json.load(open(path_b))
    if len(a) != len(b):
        print(f"MISMATCH: {len(a)} vs {len(b)} games")
        return 1
    bad = 0
    for x, y in zip(a, b):
        key = f"{x['agent']}-r{x['radius']}f{x['factions']}s{x['seed']}"
        if x != y:
            bad += 1
            if bad <= 10:
                print(f"DIFFER {key}")
                for field in ("winner", "turns", "vp"):
                    if x[field] != y[field]:
                        print(f"    {field}: old={x[field]}  new={y[field]}")
    print(f"{len(a) - bad}/{len(a)} games identical")
    return 1 if bad else 0


if __name__ == "__main__":
    if sys.argv[1] == "diff":
        sys.exit(diff(sys.argv[2], sys.argv[3]))
    run(sys.argv[1], MATRIX_M6A if "--m6a" in sys.argv else None)
