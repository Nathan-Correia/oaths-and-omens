"""
Hex-grid geometry (cube coordinates), self-contained so the engine
package doesn't depend on the visualizer's hex_common.py.
"""

from functools import lru_cache

CUBE_DIRECTIONS = [
    (1, 0, -1), (1, -1, 0), (0, -1, 1),
    (-1, 0, 1), (-1, 1, 0), (0, 1, -1),
]


def cube_hexes_in_radius(radius):
    coords = []
    for q in range(-radius, radius + 1):
        r1 = max(-radius, -q - radius)
        r2 = min(radius, -q + radius)
        for r in range(r1, r2 + 1):
            s = -q - r
            coords.append((q, r, s))
    return coords


@lru_cache(maxsize=None)
def hex_neighbors(coord, radius):
    """Cached: adjacency for a fixed (coord, radius) never changes over
    a game, and this gets called extremely often (once per eligible
    army per step, from multiple agents/phases) - profiling showed it
    as one of the hottest functions in the engine. Total distinct
    (coord, radius) pairs ever queried is bounded by the board size, so
    the cache stays small (unbounded maxsize is safe here).

    Returns a list - callers must treat it as read-only (iterate, don't
    mutate), since the same list object is handed out to every caller."""
    q, r, s = coord
    out = []
    for dq, dr, ds in CUBE_DIRECTIONS:
        nq, nr, ns = q + dq, r + dr, s + ds
        if max(abs(nq), abs(nr), abs(ns)) <= radius:
            out.append((nq, nr, ns))
    return out


@lru_cache(maxsize=None)
def hex_distance(a, b):
    """Cached for the same reason as hex_neighbors: pure function of two
    coords, no board state involved, and it's called very often (agent
    pathing heuristics, nearest-city search) with plenty of repeated
    (a, b) pairs across a turn. Distinct pairs are bounded by board size
    squared, which stays small even for the largest supported maps."""
    return max(abs(a[0] - b[0]), abs(a[1] - b[1]), abs(a[2] - b[2]))