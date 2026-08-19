"""
Shared hex-grid geometry and constants.

Used by both map_generator.py (which only needs the coordinate math)
and hex_visualizer.py (which also needs pixel conversion for drawing).

Coordinate system: flat-top hexes, cube coordinates (q, r, s) with
q + r + s == 0.
"""

import math

RADIUS = 8  # hex-grid radius -> 17 hexes across the diameter, 217 total
NUM_FACTIONS = 8

TERRAIN_TYPES = ["hills", "plains", "forest", "ocean", "mountain"]

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


def hex_to_pixel(q, r, size):
    x = size * (3 / 2 * q)
    y = size * (math.sqrt(3) / 2 * q + math.sqrt(3) * r)
    return x, y


def hex_corner(center, size, i):
    angle_rad = math.pi / 180 * (60 * i)
    return (center[0] + size * math.cos(angle_rad),
            center[1] + size * math.sin(angle_rad))


def compute_hex_size(radius, window_w, window_h, margin):
    avail_w = window_w - 2 * margin
    avail_h = window_h - 2 * margin
    board_width_units = 1.5 * (2 * radius) + 2
    board_height_units = math.sqrt(3) * (2 * radius + 1)
    size_from_w = avail_w / board_width_units
    size_from_h = avail_h / board_height_units
    return max(8, min(size_from_w, size_from_h))


def hex_neighbors(coord, radius):
    """Valid in-bounds neighbors of a cube coord."""
    q, r, s = coord
    out = []
    for dq, dr, ds in CUBE_DIRECTIONS:
        nq, nr, ns = q + dq, r + dr, s + ds
        if max(abs(nq), abs(nr), abs(ns)) <= radius:
            out.append((nq, nr, ns))
    return out


def hex_distance(a, b):
    return max(abs(a[0] - b[0]), abs(a[1] - b[1]), abs(a[2] - b[2]))