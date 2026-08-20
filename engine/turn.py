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

# Labels for the 10 checkpoints a logged turn produces (keyframe + 9
# phase deltas): used by the visualizer's intra-turn slider.
CHECKPOINT_LABELS = ["Start", "Buy", "Move 1", "Move 2", "Move 3",
                      "Cav 1", "Cav 2", "Cav 3", "Cav 4", "Battle"]


def snapshot_hexes(state):
    """Full per-hex snapshot (all coords), combining board + any
    pending-battle contributions, for logging/visualization. Terrain
    is intentionally omitted - it's static for the whole game and gets
    factored out into the log file's top-level terrain map instead."""
    result = []
    for coord, h in state.board.items():
        entry = {"q": coord[0], "r": coord[1], "s": coord[2],
                  "city": h.city_owner, "troops": None, "battle": None}
        if h.locked:
            battle = state.battles.get(coord)
            if battle is not None:
                totals = battle.faction_totals()
                entry["battle"] = {
                    "contributions": [
                        {"faction": f, "infantry": t["infantry"], "cavalry": t["cavalry"], "archers": t["archers"]}
                        for f, t in totals.items()
                    ]
                }
        else:
            entry["troops"] = dict(h.army) if h.army is not None else None
        result.append(entry)
    return result


def sparse_hexes(full_hexes):
    """Filters a snapshot down to only hexes with something on them
    (city, troops, or a pending battle) - used for the once-per-turn
    keyframe, since most hexes most of the time are just empty terrain."""
    return [e for e in full_hexes if e["city"] is not None or e["troops"] is not None or e["battle"] is not None]


def diff_hexes(before_full, after_full):
    """Compares two full snapshots (same coord order) and returns only
    the hex entries that actually changed - used for each phase's delta."""
    changed = []
    for before, after in zip(before_full, after_full):
        if before != after:
            changed.append(after)
    return changed


def get_legal_target_actions(battle, faction):
    """Valid targets for `faction` this round: any other faction still
    alive in the battle."""
    totals = battle.faction_totals()
    return [f for f, t in totals.items() if f != faction and sum(t.values()) > 0]


def _run_battle_phase(state, agents, rng):
    """Resolves every pending battle. Returns a list of per-battle event
    logs (contributions at battle start, archer phase, every round's
    detail, winner, and rectification choice) - always computed (it's
    cheap, bounded by actual rounds fought) even if the caller doesn't
    end up persisting it."""
    battle_events = []
    pending_hexes = list(state.battles.keys())
    for hex_coord in pending_hexes:
        battle = state.battles.get(hex_coord)
        if battle is None:
            continue

        contributions_start = [dict(c) for c in battle.contributions]

        def target_fn(battle, faction, _hex=hex_coord):
            agent = agents[faction]
            legal = get_legal_target_actions(battle, faction)
            return agent.decide_target(state, battle, faction, legal)

        full_log = resolve_full_battle(battle, target_fn, state, rng)

        winner = get_winner(battle)
        send_back = []
        if winner is None:
            # mutual annihilation - clear the hex, no rectification needed
            h = state.board[hex_coord]
            h.army = None
            h.locked = False
            del state.battles[hex_coord]
        else:
            agent = agents[winner]
            send_back = agent.decide_rectification(state, battle, winner)
            rectify_overflow(state, hex_coord, winner, send_back)

        battle_events.append({
            "hex": list(hex_coord),
            "contributions_start": contributions_start,
            "archer_phase": full_log["archer_phase"],
            "rounds": full_log["rounds"],
            "winner": winner,
            "rectification": send_back,
        })

    return battle_events


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


def run_turn_and_log(state, agents, rng=None):
    """Same as run_turn, but also builds and returns a turn_record
    capturing everything needed to reconstruct and inspect this turn
    later: a sparse keyframe (state at turn start), the actions each
    agent actually chose at every phase (never the legal-action menu -
    only real choices, kept sparse by omitting empty ones), and the
    resulting board deltas for each phase.

    Returns (state, turn_record). turn_record shape:
      {
        "turn_number": int,
        "keyframe": sparse_hexes(...) at turn start,
        "actions": {"buy": {...}, "movement": [...], "cavalry": [...]},
        "battle_events": [...],
        "deltas": {"buy": [...], "movement": [...], "cavalry": [...], "battle": [...]},
      }
    """
    rng = rng or random.Random()

    turn_number = state.turn_number
    keyframe = sparse_hexes(snapshot_hexes(state))

    state = apply_income_phase(state)

    before = snapshot_hexes(state)
    buy_actions = {}
    for faction, agent in agents.items():
        if not state.players[faction].alive:
            continue
        legal = get_legal_buy_actions(state, faction)
        chosen = agent.decide_buy(state, faction, legal)
        if chosen:
            buy_actions[faction] = chosen
    state = apply_buy_phase(state, buy_actions)
    after = snapshot_hexes(state)
    buy_delta = diff_hexes(before, after)

    movement_actions = []
    movement_deltas = []
    for step in range(MOVEMENT_STEPS):
        before = snapshot_hexes(state)
        actions = {}
        for faction, agent in agents.items():
            if not state.players[faction].alive:
                continue
            legal = get_legal_movement_actions(state, faction)
            chosen = agent.decide_movement(state, faction, step, legal)
            if chosen:
                actions[faction] = chosen
        apply_movement_step(state, actions)
        after = snapshot_hexes(state)
        movement_actions.append(actions)
        movement_deltas.append(diff_hexes(before, after))

    cavalry_actions = []
    cavalry_deltas = []
    for step in range(CAVALRY_STEPS):
        before = snapshot_hexes(state)
        actions = {}
        for faction, agent in agents.items():
            if not state.players[faction].alive:
                continue
            legal = get_legal_cavalry_actions(state, faction)
            chosen = agent.decide_cavalry(state, faction, step, legal)
            if chosen:
                actions[faction] = chosen
        apply_movement_step(state, actions)
        after = snapshot_hexes(state)
        cavalry_actions.append(actions)
        cavalry_deltas.append(diff_hexes(before, after))

    before = snapshot_hexes(state)
    battle_events = _run_battle_phase(state, agents, rng)
    state = apply_terrain_effects(state)
    _update_elimination(state)
    after = snapshot_hexes(state)
    battle_delta = diff_hexes(before, after)

    state.turn_number += 1

    turn_record = {
        "turn_number": turn_number,
        "keyframe": keyframe,
        "actions": {
            "buy": buy_actions,
            "movement": movement_actions,
            "cavalry": cavalry_actions,
        },
        "battle_events": battle_events,
        "deltas": {
            "buy": buy_delta,
            "movement": movement_deltas,
            "cavalry": cavalry_deltas,
            "battle": battle_delta,
        },
    }
    return state, turn_record


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