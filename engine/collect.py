"""
Collect phase.

TRANSITIONAL SHIM (PLAN.md §1.2, §5). Re-exports the C++ engine so existing
callers - run.py, tournament.py and all twelve agents - keep working with
`from engine.collect import ...` unchanged. Deleted at M8 along with the bindings.

The frozen Python reference these replace is in engine/engine_old/.
"""

from oo_engine import (  # noqa: F401
    OUTPOST_DESTROY_VP,
    VP_TO_WIN,
    apply_collect_phase,
    apply_gold_income,
    apply_resource_income,
    apply_victory_points,
)

CAPITAL_GOLD = 3
OUTPOST_GOLD = 1
OUTPOST_GOLD_WITH_BARRACKS = 2
OUTPOST_VP_PER_ROUND = 1
TEMPLE_VP_PER_ROUND = 1
