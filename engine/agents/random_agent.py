"""
Random agent. Each decision uniformly samples from the legal-action
list the engine already computed. Its main value isn't playing well -
it's exercising every code path in the engine (fuzzing) to surface
crashes, illegal states, and rules ambiguities early.
"""

import random

from .base_agent import BaseAgent
from ..state import MAX_STACK_SIZE


class RandomAgent(BaseAgent):
    def __init__(self, faction, rng=None):
        super().__init__(faction)
        self.rng = rng or random.Random()

    def decide_buy(self, state, faction, legal_actions):
        if not legal_actions:
            return []
        # take a handful of random legal purchases rather than just one,
        # so buy phases actually spend down silver over a game
        n = self.rng.randint(0, min(3, len(legal_actions)))
        return self.rng.sample(legal_actions, n) if n > 0 else []

    def decide_movement(self, state, faction, step, legal_actions):
        if not legal_actions:
            return []
        # move zero or one army per step at random (submitting moves for
        # every possible split every step would be needlessly chaotic)
        if self.rng.random() < 0.5:
            return []
        return [self.rng.choice(legal_actions)]

    def decide_cavalry(self, state, faction, step, legal_actions):
        if not legal_actions:
            return []
        if self.rng.random() < 0.5:
            return []
        return [self.rng.choice(legal_actions)]

    def decide_target(self, state, battle, faction, legal_targets):
        if not legal_targets:
            return None
        return self.rng.choice(legal_targets)

    def decide_rectification(self, state, battle, winning_faction):
        totals = battle.faction_totals()[winning_faction]
        total_units = sum(totals.values())
        overflow = total_units - MAX_STACK_SIZE
        if overflow <= 0:
            return []

        origins = [c["origin_hex"] for c in battle.contributions if c["faction"] == winning_faction]
        if not origins:
            return []

        send_back = []
        remaining = dict(totals)
        for ut in ("infantry", "cavalry", "archers"):
            while overflow > 0 and remaining[ut] > 0:
                origin = self.rng.choice(origins)
                send_back.append({"origin_hex": origin, "units": {ut: 1}})
                remaining[ut] -= 1
                overflow -= 1
        return send_back
