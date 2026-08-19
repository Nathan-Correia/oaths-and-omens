"""
Abstract agent interface. Every concrete agent (random, scripted, RL)
implements this same interface, which is what keeps engine logic fully
decoupled from agent logic - engine/turn.py never knows or cares which
kind of agent it's talking to.

Each decide_* method receives the already-computed legal-action list
for that decision point (the engine did the legality bookkeeping), and
should return actions in the shape apply_*_phase / apply_movement_step
expect. Returning an empty list / None is always legal (pass).
"""


class BaseAgent:
    def __init__(self, faction):
        self.faction = faction

    def decide_buy(self, state, faction, legal_actions):
        """Returns a list of buy actions to attempt this buy phase."""
        raise NotImplementedError

    def decide_movement(self, state, faction, step, legal_actions):
        """Returns a list of MoveActions for this movement step."""
        raise NotImplementedError

    def decide_cavalry(self, state, faction, step, legal_actions):
        """Returns a list of MoveActions for this cavalry-phase step."""
        raise NotImplementedError

    def decide_target(self, state, battle, faction, legal_targets):
        """Returns a single target faction id, or None to not attack this round."""
        raise NotImplementedError

    def decide_rectification(self, state, battle, winning_faction):
        """Returns a list of {"origin_hex": coord, "units": {...}} describing
        how to send overflow units (beyond the 6-stack cap) back to the
        winner's own contributing origin hexes. May return [] if no
        overflow needs handling (rectify_overflow is a no-op then)."""
        raise NotImplementedError
