"""
RandomAgent for engine - ported from engine/agents/random_agent.py.
Uniformly samples from whatever legal actions/mask the engine already
computed. Its value isn't playing well - it's exercising every code path
in the engine (fuzzing) to surface crashes, illegal states, and rules
ambiguities early.

Unlike v1's RandomAgent (which sub-samples a random partial split of a
chosen move), engine's action representation only ever moves a fixed,
phase-determined subset of a hex's army - see movement.py's SCOPE
decision - so there's no split left to sample: decide_movement/
decide_cavalry just returns one (hex_index, direction) pulled from the
legal mask, or None.

decide_target/decide_rectification are reused as-is by
smart_random_agent.py and heuristic_agent.py, mirroring v1's
SmartRandomAgent/HeuristicAgent inheriting these two unchanged from
RandomAgent - neither battle targeting nor rectification needs to be
smart.
"""

import random

import numpy as np

from engine.battle import faction_totals, get_legal_target_actions
from engine.state import MAX_STACK_SIZE

SKIP_CHANCE = 0.5  # chance to move nothing this step, matching v1's flavor


def random_movement(rng, legal_mask):
    if rng.random() < SKIP_CHANCE:
        return None
    rows, cols = np.nonzero(legal_mask)
    if len(rows) == 0:
        return None
    i = rng.randrange(len(rows))
    return int(rows[i]), int(cols[i])


def random_target(state, hex_index, faction, rng):
    legal = get_legal_target_actions(state, hex_index, faction)
    if not legal:
        return None
    return rng.choice(legal)


def random_rectification(state, hex_index, winner_faction, rng):
    totals = faction_totals(state, hex_index)[winner_faction]
    overflow = int(totals.sum()) - MAX_STACK_SIZE
    if overflow <= 0:
        return []
    origins = [
        int(state.battle_origin[hex_index, k]) for k in range(state.battle_faction.shape[1])
        if state.battle_faction[hex_index, k] == winner_faction
    ]
    if not origins:
        return []
    send_back = []
    remaining = [int(x) for x in totals]
    for ut in (0, 1, 2):
        while overflow > 0 and remaining[ut] > 0:
            units = [0, 0, 0]
            units[ut] = 1
            send_back.append({"origin_hex": rng.choice(origins), "units": units})
            remaining[ut] -= 1
            overflow -= 1
    return send_back


def random_buy(legal, rng, max_actions=3):
    if not legal:
        return []
    n = rng.randint(0, min(max_actions, len(legal)))
    return rng.sample(legal, n) if n > 0 else []


def make_random_agents(num_factions, seed=0):
    """Returns (decide_buy, decide_movement, decide_cavalry, decide_target,
    decide_rectification) - each {faction: callable}, matching
    engine.turn.run_turn's expected signatures. Each faction gets its
    own random.Random (mirrors v1's per-agent rng), keyed off `seed` so a
    whole game's agent decisions are reproducible."""
    rngs = {f: random.Random(seed * 1_000_003 + f) for f in range(num_factions)}

    def decide_buy(state, faction, legal):
        return random_buy(legal, rngs[faction])

    def decide_movement(state, faction, step, legal_mask):
        return random_movement(rngs[faction], legal_mask)

    def decide_cavalry(state, faction, step, legal_mask):
        return random_movement(rngs[faction], legal_mask)

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
