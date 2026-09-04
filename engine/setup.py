"""
Initial state and terrain generation.

TRANSITIONAL SHIM (PLAN.md §1.2, §5). Re-exports the C++ engine so existing
callers - run.py, tournament.py and all twelve agents - keep working with
`from engine.setup import ...` unchanged. Deleted at M8 along with the bindings.

The frozen Python reference these replace is in engine/engine_old/.
"""

from oo_engine import (  # noqa: F401
    STARTING_GOLD,
    STARTING_KILL_XP,
    create_initial_state,
)

BAG_COUNTS = {
    "plains": 120,
    "lake": 25,
    "mountain": 25,
    "desert": 40,
    "marsh": 40,
}
