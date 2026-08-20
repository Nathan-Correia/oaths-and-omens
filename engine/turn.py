"""
Top-level turn orchestrator. This is the only file that knows the full
phase sequence; every other engine file is agent-agnostic (pure
functions over plain data). Agents plug in via the interface defined
in agents/base_agent.py.
"""

import random

from .income import apply_income_phase
from .buy import get_legal_buy_actions, apply_buy_phase
from .movement import get_legal_movement_actions, get_legal_cavalry_actions, apply_movement_step
from .battle import resolve_full_battle, get_winner, rectify_overflow
from .terrain import apply_terrain_effects
from .state import army_total

MOVEMENT_STEPS = 3
CAVALRY_STEPS = 4
DEFAULT_MAX_TURNS = 100


def get_legal_target_actions(battle, faction):
    """Valid targets for `faction` this round: any other faction still
    alive in the battle."""
    totals = battle.faction_totals()
    return [f for f, t in totals.items() if f != faction and sum(t.values()) > 0]


def _run_battle_phase(state, agents, rng):
    pending_hexes = list(state.battles.keys())
    for hex_coord in pending_hexes:
        battle = state.battles.get(hex_coord)
        if battle is None:
            continue

        def target_fn(battle, faction, _hex=hex_coord):
            agent = agents[faction]
            legal = get_legal_target_actions(battle, faction)
            return agent.decide_target(state, battle, faction, legal)

        resolve_full_battle(battle, target_fn, state, rng)

        winner = get_winner(battle)
        if winner is None:
            # mutual annihilation - clear the hex, no rectification needed
            h = state.board[hex_coord]
            h.army = None
            h.locked = False
            del state.battles[hex_coord]
            continue

        agent = agents[winner]
        send_back = agent.decide_rectification(state, battle, winner)
        rectify_overflow(state, hex_coord, winner, send_back)


def _update_elimination(state):
    for faction, player in state.players.items():
        if not player.alive:
            continue
        has_city = any(h.city_owner == faction for h in state.board.values())
        has_units = any(h.army and h.army["faction"] == faction for h in state.board.values())
        if not has_city and not has_units:
            player.alive = False


def run_turn(state, agents, rng=None, on_step=None):
    """agents: {faction_id: BaseAgent}. Mutates and returns `state`.

    on_step: optional callback, called with `state` after every
    movement and cavalry step (not after income/buy/battle/terrain -
    v1 only needs movement visualized, see conversation). Used for
    replay logging without duplicating this function's phase loop."""
    rng = rng or random.Random()

    state = apply_income_phase(state)

    buy_actions = {}
    for faction, agent in agents.items():
        if not state.players[faction].alive:
            continue
        legal = get_legal_buy_actions(state, faction)
        buy_actions[faction] = agent.decide_buy(state, faction, legal)
    state = apply_buy_phase(state, buy_actions)

    for step in range(MOVEMENT_STEPS):
        actions = {}
        for faction, agent in agents.items():
            if not state.players[faction].alive:
                continue
            legal = get_legal_movement_actions(state, faction)
            actions[faction] = agent.decide_movement(state, faction, step, legal)
        apply_movement_step(state, actions)
        if on_step:
            on_step(state)

    for step in range(CAVALRY_STEPS):
        actions = {}
        for faction, agent in agents.items():
            if not state.players[faction].alive:
                continue
            legal = get_legal_cavalry_actions(state, faction)
            actions[faction] = agent.decide_cavalry(state, faction, step, legal)
        apply_movement_step(state, actions)
        if on_step:
            on_step(state)

    _run_battle_phase(state, agents, rng)
    state = apply_terrain_effects(state)
    _update_elimination(state)

    state.turn_number += 1
    return state


def check_game_end(state, max_turns=DEFAULT_MAX_TURNS):
    alive = [f for f, p in state.players.items() if p.alive]
    if len(alive) <= 1:
        return True
    return state.turn_number >= max_turns


def tally_final_score(state):
    """3-category score: cities, votes (stubbed - deferred, see convo),
    military (3/2/1 points for top 3 remaining unit counts)."""
    scores = {f: 0 for f in state.players}

    for faction in state.players:
        cities = sum(1 for h in state.board.values() if h.city_owner == faction)
        scores[faction] += cities

    # voting deferred for v1 - every player's vote total contributes 0

    unit_counts = {}
    for faction in state.players:
        total = sum(army_total(h.army) for h in state.board.values() if h.army and h.army["faction"] == faction)
        unit_counts[faction] = total
    ranked = sorted(unit_counts.items(), key=lambda kv: -kv[1])
    military_points = [3, 2, 1]
    for i, (faction, _count) in enumerate(ranked[:3]):
        scores[faction] += military_points[i]

    return scores