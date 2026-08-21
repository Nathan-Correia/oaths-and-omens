"""
Array-based hex geometry for engine_v2.

engine/geometry.py represents hexes as (q, r, s) tuples and looks up
adjacency via a dict-backed cache. engine_v2 instead assigns every hex a
fixed integer index (0..num_hexes-1) for a given board radius, and
precomputes a [num_hexes, 6] neighbor table once. All of engine_v2's game
state is then just numpy arrays indexed by that same integer, which is
what makes it array/tensor-friendly (and eventually GPU-batchable) instead
of dict-of-objects-friendly.

The index assignment order matters for cross-checking against engine/:
cube_hexes_in_radius below is copied verbatim from engine/geometry.py so
that index i always corresponds to the same (q, r, s) coordinate a v1
GameState would use, letting engine_v2.state's v1<->v2 converters line up
hex-for-hex.
"""

import numpy as np

CUBE_DIRECTIONS = [
    (1, 0, -1), (1, -1, 0), (0, -1, 1),
    (-1, 0, 1), (-1, 1, 0), (0, 1, -1),
]
NUM_DIRECTIONS = 6


def cube_hexes_in_radius(radius):
    """Identical enumeration order to engine/geometry.py's version of
    this function - this order IS the index assignment used everywhere
    else in engine_v2, so it must match exactly."""
    coords = []
    for q in range(-radius, radius + 1):
        r1 = max(-radius, -q - radius)
        r2 = min(radius, -q + radius)
        for r in range(r1, r2 + 1):
            s = -q - r
            coords.append((q, r, s))
    return coords


class HexGrid:
    """Fixed hex-index bookkeeping for one board radius.

    coord_to_index / coords: the index<->coordinate mapping.
    neighbor_table: int32[num_hexes, 6]. neighbor_table[i, d] is the
    index of hex i's neighbor in direction CUBE_DIRECTIONS[d], or -1 if
    that direction falls off the board. This is the array-table
    equivalent of engine/geometry.py's cached hex_neighbors() - built
    once per radius (never changes over a game) and then just indexed,
    no per-call work.
    """

    def __init__(self, radius):
        self.radius = radius
        self.coords = cube_hexes_in_radius(radius)
        self.num_hexes = len(self.coords)
        self.coord_to_index = {c: i for i, c in enumerate(self.coords)}

        neighbor_table = np.full((self.num_hexes, NUM_DIRECTIONS), -1, dtype=np.int32)
        for i, (q, r, s) in enumerate(self.coords):
            for d, (dq, dr, ds) in enumerate(CUBE_DIRECTIONS):
                cand = (q + dq, r + dr, s + ds)
                j = self.coord_to_index.get(cand)
                if j is not None:
                    neighbor_table[i, d] = j
        self.neighbor_table = neighbor_table

    def index_of(self, coord):
        return self.coord_to_index[tuple(coord)]

    def coord_of(self, index):
        return self.coords[index]

    def direction_between(self, from_index, to_index):
        """Which of the 6 directions leads from from_index to to_index,
        or None if they aren't neighbors. Only used by tests/tools that
        need to translate a v1 (from_hex, to_hex) move into v2's
        (hex_index, direction) action form."""
        row = self.neighbor_table[from_index]
        matches = np.nonzero(row == to_index)[0]
        return int(matches[0]) if len(matches) else None
