"""
Torch-based hex geometry for engine.

Board TOPOLOGY (neighbor_table, coords_array) is never batched - every game
in one batch shares the same radius, so the same neighbor/coordinate tables
apply to all of them. Only game STATE (engine/state.py's ArrayState) carries
a batch dimension. hex_distance/cube_hexes_in_radius/min_hex_distance_to_any
work on plain coordinates or small unbatched tensors and are used both by
per-game CPU setup code (engine/setup.py, engine/placement.py) and by the
batched turn-loop phases for board-geometry lookups that don't vary by
batch item.

The index assignment order matters for cross-checking against engine_old/:
cube_hexes_in_radius below is copied verbatim from the original geometry.py
so that index i always corresponds to the same (q, r, s) coordinate the
pre-rewrite engine used, letting differential tests line up hex-for-hex.
"""

import torch

CUBE_DIRECTIONS = [
    (1, 0, -1), (1, -1, 0), (0, -1, 1),
    (-1, 0, 1), (-1, 1, 0), (0, 1, -1),
]
NUM_DIRECTIONS = 6


def cube_hexes_in_radius(radius):
    """Identical enumeration order to the pre-rewrite engine/geometry.py's
    version of this function - this order IS the index assignment used
    everywhere else in engine, so it must match exactly (see module
    docstring)."""
    coords = []
    for q in range(-radius, radius + 1):
        r1 = max(-radius, -q - radius)
        r2 = min(radius, -q + radius)
        for r in range(r1, r2 + 1):
            s = -q - r
            coords.append((q, r, s))
    return coords


def hex_distance(a, b):
    """Cube-coordinate hex distance on plain (q, r, s) tuples/sequences -
    unchanged by the torch rewrite (setup.py/placement.py's per-game CPU
    logic and any one-off coordinate math still just wants a scalar)."""
    return max(abs(a[0] - b[0]), abs(a[1] - b[1]), abs(a[2] - b[2]))


def min_hex_distance_to_any(coords_array, ref_indices):
    """[num_hexes] int - hex distance from EVERY hex to whichever of
    coords_array[ref_indices] is nearest, computed in one vectorized pass.
    Unbatched (board-geometry-only) - see engine/buy.py's buy-phase
    legality masking for the batched version of "distance to nearest
    reference hex(es)", which has to handle a per-batch-item-varying
    reference set. ref_indices must be non-empty."""
    refs = coords_array[ref_indices]  # [k, 3]
    diff = (coords_array[:, None, :] - refs[None, :, :]).abs()  # [num_hexes, k, 3]
    return diff.amax(dim=2).amin(dim=1)


class HexGrid:
    """Fixed hex-index bookkeeping for one board radius.

    coord_to_index / coords: the index<->coordinate mapping (plain Python
    dict/list - one-off lookups, no reason to be a tensor).
    neighbor_table: int32 tensor[num_hexes, 6]. neighbor_table[i, d] is the
    index of hex i's neighbor in direction CUBE_DIRECTIONS[d], or -1 if
    that direction falls off the board. Built once per radius (never
    changes over a game, or across games in the same batch) and then just
    indexed, no per-call work.
    coords_array: int32 tensor[num_hexes, 3], the same data as `coords`
    but as one tensor, so distance from every hex to a reference
    coordinate (or set of them) can be computed with one vectorized op
    instead of a Python loop calling hex_distance per hex.
    device: which torch device neighbor_table/coords_array live on - the
    batched turn-loop wants these on the same device as game state (GPU
    if training there); per-game CPU setup code just uses the default
    (CPU).
    """

    def __init__(self, radius, device=None):
        self.radius = radius
        self.device = torch.device(device) if device is not None else torch.device("cpu")
        self.coords = cube_hexes_in_radius(radius)
        self.num_hexes = len(self.coords)
        self.coord_to_index = {c: i for i, c in enumerate(self.coords)}
        self.coords_array = torch.tensor(self.coords, dtype=torch.int32, device=self.device)

        neighbor_table = torch.full((self.num_hexes, NUM_DIRECTIONS), -1, dtype=torch.int32)
        for i, (q, r, s) in enumerate(self.coords):
            for d, (dq, dr, ds) in enumerate(CUBE_DIRECTIONS):
                cand = (q + dq, r + dr, s + ds)
                j = self.coord_to_index.get(cand)
                if j is not None:
                    neighbor_table[i, d] = j
        self.neighbor_table = neighbor_table.to(self.device)

    def index_of(self, coord):
        return self.coord_to_index[tuple(coord)]

    def coord_of(self, index):
        return self.coords[index]

    def is_edge(self, index):
        """True iff hex `index` sits on the outer ring of the board - the
        max absolute cube coordinate equals the board radius. Used by
        setup.py's terrain generation (picks an edge hex to start from)
        and placement.py's edge-tile placement ban."""
        return max(abs(c) for c in self.coords[index]) == self.radius

    def direction_between(self, from_index, to_index):
        """Which of the 6 directions leads from from_index to to_index,
        or None if they aren't neighbors. Only used by tests/tools that
        need to translate a v1 (from_hex, to_hex) move into engine's
        (hex_index, direction) action form."""
        row = self.neighbor_table[from_index]
        matches = torch.nonzero(row == to_index, as_tuple=False)
        return int(matches[0]) if len(matches) else None

    def to(self, device):
        """A new HexGrid sharing this one's coords/coord_to_index but with
        neighbor_table/coords_array moved to `device` - used to move a
        CPU-built grid (from per-game setup) onto the GPU for the batched
        turn-loop, without re-deriving the topology."""
        moved = HexGrid.__new__(HexGrid)
        moved.radius = self.radius
        moved.device = torch.device(device)
        moved.coords = self.coords
        moved.num_hexes = self.num_hexes
        moved.coord_to_index = self.coord_to_index
        moved.coords_array = self.coords_array.to(device)
        moved.neighbor_table = self.neighbor_table.to(device)
        return moved
