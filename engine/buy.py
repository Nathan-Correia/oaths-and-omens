"""
Buy phase.

TRANSITIONAL SHIM (PLAN.md §1.2, §5). Re-exports the C++ engine so existing
callers - run.py, tournament.py and all twelve agents - keep working with
`from engine.buy import ...` unchanged. Deleted at M8 along with the bindings.

The frozen Python reference these replace is in engine/engine_old/.
"""

from oo_engine import (  # noqa: F401
    INFANTRY_COST,
    OUTPOST_CAP,
    OUTPOST_COST,
    _can_build_outpost,
    _outpost_count,
    apply_buy_phase,
    eligible_outpost_mask,
    get_legal_buy_actions,
)

CAVALRY = 1
ARCHERS = 2
OUTPOST_MIN_DIST_OWN_CAPITAL = 3
OUTPOST_MIN_DIST_ENEMY_CAPITAL = 2
OUTPOST_MIN_DIST_OTHER_OUTPOST = 2
UNIT_TYPE_INDEX = {"infantry": 0, "cavalry": 1, "archers": 2}

UPGRADE_COSTS = {
    "barracks": {"fish": 2, "wood": 4},
    "workshop": {"iron": 2, "clay": 2, "wood": 4},
    "temple": {"fish": 2, "iron": 2, "clay": 2, "wood": 4},
}
