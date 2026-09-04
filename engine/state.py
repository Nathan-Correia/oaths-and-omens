"""
Array-based game state.

TRANSITIONAL SHIM (PLAN.md §1.2, §5). Re-exports the C++ engine so existing
callers - run.py, tournament.py and all twelve agents - keep working with
`from engine.state import ...` unchanged. Deleted at M8 along with the bindings.

The frozen Python reference these replace is in engine/engine_old/.
"""

import numpy as np

from oo_engine import (  # noqa: F401
    MAX_BATTLE_CONTRIB,
    MAX_STACK_SIZE,
    NO_FACTION,
    NO_ORIGIN,
    NO_UPGRADE,
    ArrayState,
    count_all_units_in_play,
    count_units_in_play,
)

# Type orders. These MUST match include/oo/config.hpp's enums - the C++ side is
# authoritative, and these are the names for the same indices.
TERRAIN_TYPES = ["plains", "mountain", "lake", "desert", "marsh"]
TERRAIN_TO_INDEX = {t: i for i, t in enumerate(TERRAIN_TYPES)}
IMPASSABLE_TERRAIN_INDICES = np.array(
    [TERRAIN_TO_INDEX["mountain"], TERRAIN_TO_INDEX["lake"]], dtype=np.int8
)
IMPASSABLE_BY_TERRAIN = np.zeros(len(TERRAIN_TYPES), dtype=bool)
IMPASSABLE_BY_TERRAIN[IMPASSABLE_TERRAIN_INDICES] = True

UNIT_TYPES = ["infantry", "cavalry", "archers"]
SPAWN_CAPS = np.array([24, 12, 12], dtype=np.int32)

RESOURCE_TYPES = ["wood", "iron", "clay", "fish"]
RESOURCE_TO_INDEX = {t: i for i, t in enumerate(RESOURCE_TYPES)}

UPGRADE_TYPES = ["barracks", "workshop", "temple"]
UPGRADE_TO_INDEX = {t: i for i, t in enumerate(UPGRADE_TYPES)}
