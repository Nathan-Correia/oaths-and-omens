"""
Dumps engine_old's hex geometry so the C++ HexGrid can be checked against it.

The index assignment order is the thing that actually matters: index i must mean
the same (q, r, s) in both engines, or every array in GameState is shuffled
relative to Python's and nothing else can be compared. Everything downstream in the
port rests on this file agreeing.

Usage:  python engine/tools/dump_grid_reference.py > engine/tests/data/grid_golden.txt

Format, per radius:
  GRID <radius> <num_hexes>
  COORDS   <3n> q0 r0 s0 q1 r1 s1 ...
  NEIGHBOURS <6n> ...      (-1 off-board, in CUBE_DIRECTIONS order)
  EDGE     <n> ...
  DISTSUM  <n> ...         row sums of the full distance matrix
  DISTROW  <index> <n> ... a few full rows, spot-checked exactly
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _bootstrap  # noqa: E402

_bootstrap.install()

from engine.geometry import HexGrid, hex_distance  # noqa: E402

RADII = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]


def emit(name, values):
    values = list(values)
    sys.stdout.write(f"{name} {len(values)}")
    for v in values:
        sys.stdout.write(f" {int(v)}")
    sys.stdout.write("\n")


def main():
    for radius in RADII:
        g = HexGrid(radius)
        n = g.num_hexes
        sys.stdout.write(f"GRID {radius} {n}\n")

        flat = []
        for q, r, s in g.coords:
            flat += [q, r, s]
        emit("COORDS", flat)
        emit("NEIGHBOURS", g.neighbor_table.reshape(-1))
        emit("EDGE", [1 if g.is_edge(i) else 0 for i in range(n)])

        # Row sums catch a systematic distance error cheaply across the whole
        # matrix; the sampled full rows below catch a localized one exactly.
        sums = []
        for i in range(n):
            ci = g.coord_of(i)
            sums.append(sum(hex_distance(ci, g.coord_of(j)) for j in range(n)))
        emit("DISTSUM", sums)

        for i in sorted({0, n // 2, n - 1, n // 3, (2 * n) // 3}):
            ci = g.coord_of(i)
            sys.stdout.write(f"DISTROW {i}")
            row = [hex_distance(ci, g.coord_of(j)) for j in range(n)]
            sys.stdout.write(f" {len(row)}")
            for v in row:
                sys.stdout.write(f" {v}")
            sys.stdout.write("\n")
        sys.stdout.write("ENDGRID\n")


if __name__ == "__main__":
    main()
