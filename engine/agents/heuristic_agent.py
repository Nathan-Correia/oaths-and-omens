"""
HeuristicAgent - plays with real tactical judgment rather than pure
greed or randomness, while keeping the parts that genuinely don't
matter much (multi-battle targeting, rectification) simple/random.

Buy phase: same aggressive spend-everything approach as SmartRandomAgent
(more units is generally better), but kill-XP conversions now balance
composition - converts toward whichever of cavalry/archers this faction
currently has fewer of, rather than a pure coin flip.

Movement / cavalry phase: for the single army it's allowed to move this
step (still one army per step - see engine/movement.py), in priority
order:
  1. ATTACK - if a weaker enemy stack is adjacent, attack the weakest one.
  2. RETREAT - if a meaningfully stronger enemy stack is adjacent and we
     have no favorable attack available, fall back toward our own
     nearest city instead of standing still to be destroyed.
  3. ADVANCE - otherwise, march toward the nearest enemy city, but never
     knowingly step onto a hex held by a stronger enemy army along the
     way (picks the next-best safe step instead).

A small amount of randomness is mixed in for flavor (see RANDOM_MOVE_CHANCE)
so games don't all look mechanically identical - occasionally this agent
just does something random instead of the "correct" heuristic move.

Battle targeting and rectification are inherited unchanged from
RandomAgent, per the instruction that this doesn't need to be smart.
"""

import random

from .smart_random_agent import SmartRandomAgent
from ..geometry import hex_neighbors, hex_distance
from ..state import IMPASSABLE_TERRAIN, army_total, count_units_in_play

RANDOM_MOVE_CHANCE = 0.12          # flavor noise: chance to ignore the heuristic and act randomly
RETREAT_THRESHOLD = 0.85           # retreat if our army is weaker than this fraction of the threat


class HeuristicAgent(SmartRandomAgent):

    def decide_buy(self, state, faction, legal_actions):
        player = state.players[faction]
        infantry_actions = [a for a in legal_actions if a["type"] == "buy_infantry"]
        convert_actions = [a for a in legal_actions if a["type"] == "convert_to_special"]

        chosen = []

        if infantry_actions:
            num_purchases = player.silver // 2  # INFANTRY_COST, avoiding an import cycle
            for _ in range(num_purchases):
                chosen.append(self.rng.choice(infantry_actions))

        if convert_actions:
            num_conversions = player.kill_xp
            # Decisions here don't mutate `state` (only apply_buy_phase does,
            # later), so these counts are constant across every iteration of
            # this loop - compute once rather than re-scanning the whole
            # board+battles per conversion (profiling showed this as a
            # significant cost with several kill-XP banked at once).
            cav_count = count_units_in_play(state, faction, "cavalry")
            arc_count = count_units_in_play(state, faction, "archers")
            for _ in range(num_conversions):
                # lean toward whichever type we have fewer of; small chance
                # to ignore that and pick randomly, for flavor
                if self.rng.random() < RANDOM_MOVE_CHANCE:
                    unit_type = self.rng.choice(["cavalry", "archers"])
                else:
                    unit_type = "cavalry" if cav_count <= arc_count else "archers"
                matching = [a for a in convert_actions if a["unit_type"] == unit_type]
                if not matching:
                    matching = convert_actions
                if matching:
                    chosen.append(self.rng.choice(matching))

        return chosen

    def _nearest_own_city(self, state, faction, from_hex):
        own_cities = [coord for coord, h in state.board.items() if h.city_owner == faction]
        if not own_cities:
            return None
        return min(own_cities, key=lambda c: hex_distance(from_hex, c))

    def _adjacent_enemy_stacks(self, state, faction, origin):
        """[(hex, enemy_total_units)] for every hostile army orthogonally
        adjacent to `origin`."""
        result = []
        for n in hex_neighbors(origin, state.radius):
            nh = state.board.get(n)
            if nh and nh.army and nh.army["faction"] != faction:
                result.append((n, army_total(nh.army)))
        return result

    def _safe_best_step_toward(self, state, faction, origin, target_hex, my_total):
        """Among passable neighbors, prefer the one that most reduces
        distance to target_hex while not being held by a strictly
        stronger enemy army. Falls back to the globally-closest neighbor
        if every option is guarded by something stronger (better to try
        than to freeze in place forever)."""
        neighbors = hex_neighbors(origin, state.radius)
        candidates = [n for n in neighbors if state.board[n].terrain not in IMPASSABLE_TERRAIN]
        if not candidates:
            return None

        def is_unsafe(n):
            nh = state.board[n]
            return nh.army is not None and nh.army["faction"] != faction and army_total(nh.army) >= my_total

        safe = [n for n in candidates if not is_unsafe(n)]
        pool = safe if safe else candidates
        return min(pool, key=lambda n: hex_distance(n, target_hex))

    def _heuristic_move(self, state, faction, legal_actions):
        if not legal_actions:
            return []

        if self.rng.random() < RANDOM_MOVE_CHANCE:
            return [self.rng.choice(legal_actions)]

        actions_by_origin = {}
        for a in legal_actions:
            actions_by_origin.setdefault(tuple(a["from_hex"]), []).append(a)

        # get_legal_movement_actions/get_legal_cavalry_actions now hand back
        # exactly one action per direction per origin, always moving the
        # whole army/cavalry contingent (see movement.py) - so every action
        # for a given origin already carries the same unit total, and the
        # old max(...) over all of an origin's actions was recomputing that
        # same sum once per direction for nothing.
        ranked_origins = sorted(
            actions_by_origin.items(),
            key=lambda item: sum(item[1][0]["units"].values()),
            reverse=True,
        )

        # City ownership can't change during this (read-only) decision - it
        # only changes via apply_movement_step/battle resolution, which run
        # later. Scan the board for city coords once per call and reuse
        # across every origin tried below, instead of _nearest_own_city/
        # _nearest_enemy_city each rescanning the whole board per origin.
        own_cities = [coord for coord, h in state.board.items() if h.city_owner == faction]
        enemy_cities = [coord for coord, h in state.board.items()
                         if h.city_owner is not None and h.city_owner != faction]

        for origin, actions in ranked_origins:
            origin_army = state.board[origin].army
            my_total = army_total(origin_army) if origin_army else 0
            if my_total <= 0:
                continue

            adjacent_enemies = self._adjacent_enemy_stacks(state, faction, origin)

            # 1. ATTACK - weakest adjacent enemy we can beat
            favorable = [(n, t) for n, t in adjacent_enemies if my_total > t]
            if favorable:
                target_hex, _ = min(favorable, key=lambda pair: pair[1])
                matching = [a for a in actions if tuple(a["to_hex"]) == target_hex]
                if matching:
                    return [max(matching, key=lambda a: sum(a["units"].values()))]

            # 2. RETREAT - a stronger enemy is adjacent and we have no
            # favorable attack; fall back toward our own nearest city
            threats = [t for _, t in adjacent_enemies if t > 0]
            if threats and my_total < max(threats) * RETREAT_THRESHOLD:
                home = min(own_cities, key=lambda c: hex_distance(origin, c)) if own_cities else None
                if home is not None:
                    best_hex = self._safe_best_step_toward(state, faction, origin, home, my_total)
                    matching = [a for a in actions if tuple(a["to_hex"]) == best_hex] if best_hex else []
                    if matching:
                        return [max(matching, key=lambda a: sum(a["units"].values()))]
                continue  # no safe retreat found - try the next-largest army instead

            # 3. ADVANCE - march toward the nearest enemy city, avoiding
            # any hex held by a strictly stronger enemy along the way
            target_city = min(enemy_cities, key=lambda c: hex_distance(origin, c)) if enemy_cities else None
            if target_city is None:
                continue
            best_hex = self._safe_best_step_toward(state, faction, origin, target_city, my_total)
            if best_hex is None:
                continue
            matching = [a for a in actions if tuple(a["to_hex"]) == best_hex]
            if matching:
                return [max(matching, key=lambda a: sum(a["units"].values()))]

        return []

    def decide_movement(self, state, faction, step, legal_actions):
        return self._heuristic_move(state, faction, legal_actions)

    def decide_cavalry(self, state, faction, step, legal_actions):
        return self._heuristic_move(state, faction, legal_actions)