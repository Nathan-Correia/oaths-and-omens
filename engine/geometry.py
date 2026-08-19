"""
Hex-grid geometry (cube coordinates), self-contained so the engine
package doesn't depend on the visualizer's hex_common.py.
"""

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


def hex_neighbors(coord, radius):
    q, r, s = coord
    out = []
    for dq, dr, ds in CUBE_DIRECTIONS:
        nq, nr, ns = q + dq, r + dr, s + ds
        if max(abs(nq), abs(nr), abs(ns)) <= radius:
            out.append((nq, nr, ns))
    return out


def hex_distance(a, b):
    return max(abs(a[0] - b[0]), abs(a[1] - b[1]), abs(a[2] - b[2]))
