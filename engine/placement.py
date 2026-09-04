"""
Capital placement and draft.

TRANSITIONAL SHIM (PLAN.md §1.2, §5). Re-exports the C++ engine so existing
callers - run.py, tournament.py and all twelve agents - keep working with
`from engine.placement import ...` unchanged. Deleted at M8 along with the bindings.

The frozen Python reference these replace is in engine/engine_old/.
"""

from oo_engine import (  # noqa: F401
    CAPITAL_MIN_DIST,
    legal_placement_mask,
    run_city_setup,
)

EDGE_BAN_MIN_FACTIONS = 5
EDGE_BAN_MAX_FACTIONS = 7
