"""
SmartRandomAgent - a step up from pure random, in exactly three ways:

  1. Buy phase: spends all available silver on infantry, and converts
     every banked kill-XP token into a cavalry or archer unit (type
     chosen randomly per token).
  2. Movement / cavalry phases: each step, the single largest eligible
     army greedily takes one step toward whichever enemy city is
     currently nearest to it (hex distance), moving the whole stack
     rather than splitting. Only one army may move per player per step
     (see engine/movement.py), so this can't move every army at once -
     it picks the biggest one each step.
  3. Battle targeting and rectification are inherited unchanged from
     RandomAgent - this agent isn't "smarter" about fighting, only
     about economy and positioning.

This is a one-step-greedy heuristic, not real pathfinding: it always
picks whichever passable neighbor hex reduces distance-to-target the
most. It has no notion of routing around impassable terrain that
blocks the direct line, so it can occasionally stall against a
mountain/lake range. Good enough as a "smarter than random" baseline;
swap in A* later if that turns out to matter in practice.
"""

from .random_agent import RandomAgent
from ..geometry import hex_neighbors, hex_distance
from ..state import IMPASSABLE_TERRAIN
from ..buy import INFANTRY_COST


class SmartRandomAgent(RandomAgent):

    def decide_buy(self, state, faction, legal_actions):
        player = state.players[faction]
        infantry_actions = [a for a in legal_actions if a["type"] == "buy_infantry"]
        convert_actions = [a for a in legal_actions if a["type"] == "convert_to_special"]

        chosen = []

        if infantry_actions:
            num_purchases = player.silver // INFANTRY_COST
            for _ in range(num_purchases):
                chosen.append(self.rng.choice(infantry_actions))

        if convert_actions:
            num_conversions = player.kill_xp
            for _ in range(num_conversions):
                unit_type = self.rng.choice(["cavalry", "archers"])
                matching = [a for a in convert_actions if a["unit_type"] == unit_type]
                if matching:
                    chosen.append(self.rng.choice(matching))

        return chosen

    def _nearest_enemy_city(self, state, faction, from_hex):
        enemy_cities = [coord for coord, h in state.board.items()
                         if h.city_owner is not None and h.city_owner != faction]
        if not enemy_cities:
            return None
        return min(enemy_cities, key=lambda c: hex_distance(from_hex, c))

    def _best_step_toward(self, state, from_hex, target_hex):
        neighbors = hex_neighbors(from_hex, state.radius)
        candidates = [n for n in neighbors if state.board[n].terrain not in IMPASSABLE_TERRAIN]
        if not candidates:
            return None
        return min(candidates, key=lambda n: hex_distance(n, target_hex))

    def _greedy_move_all_armies(self, state, faction, legal_actions):
        """Shared logic for both decide_movement and decide_cavalry: for
        every distinct origin army in legal_actions, move the whole
        stack one step toward the nearest enemy city, if possible."""
    def _greedy_move_largest_army(self, state, faction, legal_actions):
        """Only one army may move per player per step (see engine/movement.py),
        so rather than every eligible army trying to advance at once,
        this picks the single largest eligible army and moves the whole
        thing one step toward its nearest enemy city - matching "try to
        move its largest armies toward the nearest enemy city" without
        violating the one-army-per-step rule."""
        if not legal_actions:
            return []

        actions_by_origin = {}
        for a in legal_actions:
            actions_by_origin.setdefault(tuple(a["from_hex"]), []).append(a)

        # rank origins by the largest total-unit action available there
        # (a proxy for "how big is this army"), largest first
        ranked_origins = sorted(
            actions_by_origin.items(),
            key=lambda item: max(sum(a["units"].values()) for a in item[1]),
            reverse=True,
        )

        for origin, actions in ranked_origins:
            target_city = self._nearest_enemy_city(state, faction, origin)
            if target_city is None:
                continue
            best_hex = self._best_step_toward(state, origin, target_city)
            if best_hex is None:
                continue

            matching = [a for a in actions if tuple(a["to_hex"]) == best_hex]
            if not matching:
                continue

            # move the whole available stack, not a partial split
            best_action = max(matching, key=lambda a: sum(a["units"].values()))
            return [best_action]

        return []

    def decide_movement(self, state, faction, step, legal_actions):
        return self._greedy_move_largest_army(state, faction, legal_actions)

    def decide_cavalry(self, state, faction, step, legal_actions):
        return self._greedy_move_largest_army(state, faction, legal_actions)