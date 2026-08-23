"""
HeuristicAgent for engine - ported from
engine/agents/heuristic_agent.py. Plays with real tactical judgment
rather than pure greed or randomness, while keeping the parts that
genuinely don't matter much (multi-battle targeting, rectification)
simple/random (reused from random_agent.py, same as v1's inheritance).

Buy phase: same aggressive spend-everything approach as
smart_random_agent (more units is generally better), but kill-XP
conversions now balance composition - converts toward whichever of
cavalry/archers this faction currently has fewer of, rather than a pure
coin flip.

Movement / cavalry phase: for the single army it's allowed to move this
step (engine's decide_movement only ever returns one (hex_index,
direction) per faction per step - see movement.py's SCOPE decision, so
"pick the biggest army" is just which origin gets tried first, not a
separate constraint to enforce), in priority order:
  1. ATTACK - if a weaker enemy stack is adjacent, attack the weakest one.
  2. RETREAT - if a meaningfully stronger enemy stack is adjacent and we
     have no favorable attack available, fall back toward our own
     nearest city instead of standing still to be destroyed.
  3. ADVANCE - otherwise, march toward the nearest enemy city, but never
     knowingly step onto a hex held by a stronger enemy army along the
     way (picks the next-best safe step instead).

A small amount of randomness is mixed in for flavor (RANDOM_MOVE_CHANCE)
so games don't all look mechanically identical.
"""

import random

import numpy as np

from .random_agent import random_rectification, random_target
from engine.geometry import hex_distance
from engine.state import NO_FACTION, count_units_in_play

RANDOM_MOVE_CHANCE = 0.12          # flavor noise: chance to ignore the heuristic and act randomly
RETREAT_THRESHOLD = 0.85           # retreat if our army is weaker than this fraction of the threat


def heuristic_buy(state, faction, legal, rng):
    infantry_actions = [a for a in legal if a["type"] == "buy_infantry"]
    convert_actions = [a for a in legal if a["type"] == "convert_to_special"]

    chosen = []

    if infantry_actions:
        num_purchases = int(state.silver[faction]) // 2  # INFANTRY_COST, avoiding an import cycle
        for _ in range(num_purchases):
            chosen.append(rng.choice(infantry_actions))

    if convert_actions:
        num_conversions = int(state.kill_xp[faction])
        # Decisions here don't mutate `state` (only apply_buy_phase does,
        # later), so these counts are constant across every iteration of
        # this loop - compute once rather than re-scanning per conversion.
        cav_count = count_units_in_play(state, faction, 1)
        arc_count = count_units_in_play(state, faction, 2)
        for _ in range(num_conversions):
            if rng.random() < RANDOM_MOVE_CHANCE:
                unit_type = rng.choice(["cavalry", "archers"])
            else:
                unit_type = "cavalry" if cav_count <= arc_count else "archers"
            matching = [a for a in convert_actions if a["unit_type"] == unit_type]
            if not matching:
                matching = convert_actions
            if matching:
                chosen.append(rng.choice(matching))

    return chosen


def _adjacent_enemy_stacks(state, faction, origin):
    """[(direction, neighbor_index, enemy_total)] for every hostile army
    adjacent to `origin`, in ALL six directions - not just legal-move
    ones, since detecting a threat/target doesn't depend on being able
    to step there this instant."""
    grid = state.grid
    result = []
    for d in range(6):
        n = int(grid.neighbor_table[origin, d])
        if n < 0:
            continue
        if state.army_faction[n] != NO_FACTION and state.army_faction[n] != faction:
            result.append((d, n, int(state.army_units[n].sum())))
    return result


def _safe_best_direction(state, faction, origin, legal_dirs, target_coord, my_total):
    """Among `legal_dirs`, prefer whichever most reduces distance to
    target_coord while not stepping onto a hex held by a stronger enemy
    army. Falls back to the globally-closest direction if every option
    is guarded by something stronger."""
    grid = state.grid

    def is_unsafe(d):
        n = int(grid.neighbor_table[origin, d])
        return (state.army_faction[n] != NO_FACTION
                and state.army_faction[n] != faction
                and int(state.army_units[n].sum()) >= my_total)

    safe_dirs = [d for d in legal_dirs if not is_unsafe(d)]
    pool = safe_dirs if safe_dirs else legal_dirs
    return min(pool, key=lambda d: hex_distance(grid.coord_of(int(grid.neighbor_table[origin, d])), target_coord))


def heuristic_move(state, faction, legal_mask, rng):
    grid = state.grid
    origins = np.nonzero(legal_mask.any(axis=1))[0]
    if len(origins) == 0:
        return None

    if rng.random() < RANDOM_MOVE_CHANCE:
        origin = int(rng.choice(origins.tolist()))
        direction = int(rng.choice(np.nonzero(legal_mask[origin])[0].tolist()))
        return origin, direction

    sizes = state.army_units[origins].sum(axis=1)
    ranked_origins = [int(origins[i]) for i in np.argsort(-sizes)]

    own_cities = [grid.coord_of(int(i)) for i in np.nonzero(state.city_owner == faction)[0]]
    enemy_cities = [
        grid.coord_of(int(i)) for i in np.nonzero((state.city_owner != NO_FACTION) & (state.city_owner != faction))[0]
    ]

    for origin in ranked_origins:
        my_total = int(state.army_units[origin].sum())
        if my_total <= 0:
            continue

        legal_dirs = np.nonzero(legal_mask[origin])[0].tolist()
        if not legal_dirs:
            continue

        adjacent_enemies = _adjacent_enemy_stacks(state, faction, origin)

        # 1. ATTACK - weakest adjacent enemy we can beat, if reachable this step
        favorable = [(d, n, t) for d, n, t in adjacent_enemies if my_total > t]
        if favorable:
            _, _, best_t = min(favorable, key=lambda x: x[2])
            attack_dir = next((d for d, n, t in favorable if t == best_t and d in legal_dirs), None)
            if attack_dir is not None:
                return origin, attack_dir

        # 2. RETREAT - a stronger enemy is adjacent and we have no
        # favorable attack; fall back toward our own nearest city
        threats = [t for _, _, t in adjacent_enemies if t > 0]
        if threats and my_total < max(threats) * RETREAT_THRESHOLD:
            if own_cities:
                home = min(own_cities, key=lambda c: hex_distance(grid.coord_of(origin), c))
                return origin, _safe_best_direction(state, faction, origin, legal_dirs, home, my_total)
            continue  # no home city - try the next-largest army instead

        # 3. ADVANCE - march toward the nearest enemy city, avoiding any
        # hex held by a strictly stronger enemy along the way
        if not enemy_cities:
            continue
        target_city = min(enemy_cities, key=lambda c: hex_distance(grid.coord_of(origin), c))
        return origin, _safe_best_direction(state, faction, origin, legal_dirs, target_city, my_total)

    return None


def make_heuristic_agents(num_factions, seed=0):
    """Returns (decide_buy, decide_movement, decide_cavalry, decide_target,
    decide_rectification) - each {faction: callable}, matching
    engine.turn.run_turn's expected signatures."""
    rngs = {f: random.Random(seed * 1_000_003 + f) for f in range(num_factions)}

    def decide_buy(state, faction, legal):
        return heuristic_buy(state, faction, legal, rngs[faction])

    def decide_movement(state, faction, step, legal_mask):
        return heuristic_move(state, faction, legal_mask, rngs[faction])

    def decide_cavalry(state, faction, step, legal_mask):
        return heuristic_move(state, faction, legal_mask, rngs[faction])

    def decide_target(state, hex_index, faction):
        return random_target(state, hex_index, faction, rngs[faction])

    def decide_rectification(state, hex_index, winner_faction):
        return random_rectification(state, hex_index, winner_faction, rngs[winner_faction])

    factions = range(num_factions)
    return (
        {f: decide_buy for f in factions},
        {f: decide_movement for f in factions},
        {f: decide_cavalry for f in factions},
        {f: decide_target for f in factions},
        {f: decide_rectification for f in factions},
    )
