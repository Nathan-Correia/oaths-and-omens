"""
Movement.

TRANSITIONAL SHIM (PLAN.md §1.2, §5). Re-exports the C++ engine so existing
callers - run.py, tournament.py and all twelve agents - keep working with
`from engine.movement import ...` unchanged. Deleted at M8 along with the bindings.

The frozen Python reference these replace is in engine/engine_old/.
"""

from oo_engine import (  # noqa: F401
    apply_movement_step,
    legal_cavalry_mask,
    legal_movement_mask,
)

from .state import TERRAIN_TO_INDEX

MARSH_INDEX = TERRAIN_TO_INDEX["marsh"]
