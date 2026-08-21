"""
Turn orchestration for engine_v2 - ported from engine/turn.py's run_turn
(not run_turn_and_log: no replay/visualization consumer for engine_v2
yet, same reasoning as battle.py skipping the structured per-round log).

Agents aren't a formal class hierarchy here (unlike v1's BaseAgent) - each
decision point takes a plain callback instead, looked up per-faction from
a {faction: callable} dict (mirroring v1's {faction: agent}). Deliberate:
the real "agent" for engine_v2 will eventually be a neural policy with
its own natural interface (masked observation in, action out) that
almost certainly won't look like BaseAgent's five methods - building that
class hierarchy now would likely just be thrown away later. Callbacks
compose fine with an eventual class-based wrapper (an agent object's
bound methods are themselves callables) without committing to that shape
now. Callback signatures:

  decide_buy(state, faction, legal_actions) -> [action, ...]
  decide_movement(state, faction, step, legal_mask) -> (hex_index, direction) or None
  decide_cavalry(state, faction, step, legal_mask) -> (hex_index, direction) or None
  decide_target(state, hex_index, faction) -> target_faction or None
  decide_rectification(state, hex_index, winner_faction) -> [{"origin_hex", "units"}, ...]
"""

import random

import numpy as np

from .battle import get_winner, rectify_overflow, resolve_full_battle
from .buy import apply_buy_phase, get_legal_buy_actions
from .income import apply_income_phase
from .movement import apply_movement_step, legal_cavalry_mask, legal_movement_mask
from .state import NO_FACTION, NO_ORIGIN, count_units_in_play
from .terrain import apply_terrain_effects

MOVEMENT_STEPS = 3
CAVALRY_STEPS = 4
DEFAULT_MAX_TURNS = 100


def _run_battle_phase(state, decide_target, decide_rectification, rng):
    """Resolves every pending battle, in state.battle_order (battle
    creation order - see state.py's module docstring for why this has to
    match v1's dict-insertion-order semantics rather than e.g. hex-index
    order: the dismount infantry cap tally below is shared across every
    battle resolved this turn, so processing order can affect outcomes
    near the cap)."""
    infantry_counts = {f: count_units_in_play(state, f, 0) for f in range(state.num_factions)}
    pending_hexes = list(state.battle_order)

    for hex_index in pending_hexes:
        if not state.locked[hex_index]:
            continue

        def target_fn(s, hidx, faction, _faction_agent=decide_target):
            return _faction_agent[faction](s, hidx, faction)

        resolve_full_battle(state, hex_index, target_fn, rng, infantry_counts)

        winner = get_winner(state, hex_index)
        if winner is None:
            state.army_faction[hex_index] = NO_FACTION
            state.army_units[hex_index] = 0
            state.locked[hex_index] = False
            state.battle_faction[hex_index] = NO_FACTION
            state.battle_origin[hex_index] = NO_ORIGIN
            state.battle_units[hex_index] = 0
            state.battle_round[hex_index] = 0
            state.battle_order.remove(hex_index)
        else:
            send_back = decide_rectification[winner](state, hex_index, winner)
            rectify_overflow(state, hex_index, winner, send_back)


def _update_elimination(state):
    for faction in range(state.num_factions):
        if not state.alive[faction]:
            continue
        has_city = bool(np.any(state.city_owner == faction))
        has_units = bool(np.any(state.army_faction == faction))
        if not has_city and not has_units:
            state.alive[faction] = False


def run_turn(state, decide_buy, decide_movement, decide_cavalry, decide_target, decide_rectification, rng=None):
    """Mutates and returns `state`. Each decide_* argument is a
    {faction: callable} dict - see module docstring for each callable's
    signature. Only alive factions are asked for decisions, same as v1."""
    rng = rng or random.Random()

    apply_income_phase(state)

    buy_actions = {}
    for faction in range(state.num_factions):
        if not state.alive[faction]:
            continue
        legal = get_legal_buy_actions(state, faction)
        buy_actions[faction] = decide_buy[faction](state, faction, legal)
    apply_buy_phase(state, buy_actions)

    for step in range(MOVEMENT_STEPS):
        actions = {}
        for faction in range(state.num_factions):
            if not state.alive[faction]:
                continue
            legal = legal_movement_mask(state, faction)
            actions[faction] = decide_movement[faction](state, faction, step, legal)
        apply_movement_step(state, actions, cavalry_only=False)

    for step in range(CAVALRY_STEPS):
        actions = {}
        for faction in range(state.num_factions):
            if not state.alive[faction]:
                continue
            legal = legal_cavalry_mask(state, faction)
            actions[faction] = decide_cavalry[faction](state, faction, step, legal)
        apply_movement_step(state, actions, cavalry_only=True)

    _run_battle_phase(state, decide_target, decide_rectification, rng)
    apply_terrain_effects(state)
    _update_elimination(state)

    state.turn_number += 1
    return state


def check_game_end(state, max_turns=DEFAULT_MAX_TURNS):
    if int(np.sum(state.alive)) <= 1:
        return True
    return state.turn_number >= max_turns


def tally_final_score(state):
    """3-category score, matching engine/turn.py's tally_final_score:
    cities (1 pt each) + military (top 3 remaining unit counts score
    3/2/1). Voting is stubbed in v1 too (0 contribution) - see that
    module's docstring."""
    scores = {f: 0 for f in range(state.num_factions)}

    for faction in range(state.num_factions):
        scores[faction] += int(np.sum(state.city_owner == faction))

    unit_counts = {
        faction: int(state.army_units[state.army_faction == faction].sum())
        for faction in range(state.num_factions)
    }
    ranked = sorted(unit_counts.items(), key=lambda kv: -kv[1])
    military_points = [3, 2, 1]
    for i, (faction, _count) in enumerate(ranked[:3]):
        scores[faction] += military_points[i]

    return scores
