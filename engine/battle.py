"""
Battle resolution.

TRANSITIONAL SHIM (PLAN.md §1.2, §5). Re-exports the C++ engine so existing
callers - run.py, tournament.py and all twelve agents - keep working with
`from engine.battle import ...` unchanged. Deleted at M8 along with the bindings.

The frozen Python reference these replace is in engine/engine_old/.
"""

from oo_engine import (  # noqa: F401
    faction_totals,
    get_legal_target_actions,
    get_winner,
    is_battle_over,
)

DEATH_PRIORITY = (0, 1, 2)
MAX_ROUNDS_SAFETY_CAP = 50
CAPITAL_DEFENSE_SHOTS = 2
OUTPOST_DEFENSE_SHOTS = 1


def faction_alive_totals(state, hex_index):
    """{faction: total_units} for factions still alive in this battle."""
    return {f: t for f, t in faction_totals(state, hex_index).items() if int(t.sum()) > 0}
