"""
Terrain effects.

TRANSITIONAL SHIM (PLAN.md §1.2, §5). Re-exports the C++ engine so existing
callers - run.py, tournament.py and all twelve agents - keep working with
`from engine.terrain import ...` unchanged. Deleted at M8 along with the bindings.

The frozen Python reference these replace is in engine/engine_old/.
"""

from oo_engine import apply_terrain_effects  # noqa: F401

from .state import TERRAIN_TO_INDEX

DESERT_INDEX = TERRAIN_TO_INDEX["desert"]
