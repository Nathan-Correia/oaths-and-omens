"""
Turn orchestration.

TRANSITIONAL SHIM (PLAN.md §1.2, §5). Re-exports the C++ engine so existing
callers - run.py, tournament.py and all twelve agents - keep working with
`from engine.turn import ...` unchanged. Deleted at M8 along with the bindings.

The frozen Python reference these replace is in engine/engine_old/.
"""

from oo_engine import (  # noqa: F401
    CAVALRY_STEPS,
    MOVEMENT_STEPS,
    _run_battle_phase,
    check_game_end,
    get_game_winner,
    run_turn,
    tally_final_score,
)

CHECKPOINT_LABELS = ["Start", "Buy", "Move 1", "Move 2", "Move 3", "Cav 1", "Cav 2", "Battle"]
