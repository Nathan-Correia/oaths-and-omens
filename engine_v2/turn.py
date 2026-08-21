"""
Turn orchestration for engine_v2 - ported from engine/turn.py's run_turn,
plus a run_turn_and_log for replay/visualization (see this module's
run_turn_and_log docstring for how its log format differs from v1's).

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

from .battle import faction_totals, get_winner, rectify_overflow, resolve_full_battle
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
    near the cap). Returns a list of per-battle event logs, same shape
    as engine/turn.py's _run_battle_phase - always built (cheap, bounded
    by rounds actually fought), even though only run_turn_and_log ends
    up keeping it."""
    infantry_counts = {f: count_units_in_play(state, f, 0) for f in range(state.num_factions)}
    pending_hexes = list(state.battle_order)
    battle_events = []

    for hex_index in pending_hexes:
        if not state.locked[hex_index]:
            continue

        contributions_start = [
            {
                "faction": int(state.battle_faction[hex_index, k]),
                "origin_hex": int(state.battle_origin[hex_index, k]),
                "infantry": int(state.battle_units[hex_index, k, 0]),
                "cavalry": int(state.battle_units[hex_index, k, 1]),
                "archers": int(state.battle_units[hex_index, k, 2]),
            }
            for k in range(state.battle_faction.shape[1])
            if state.battle_faction[hex_index, k] != NO_FACTION
        ]

        def target_fn(s, hidx, faction, _faction_agent=decide_target):
            return _faction_agent[faction](s, hidx, faction)

        full_log = resolve_full_battle(state, hex_index, target_fn, rng, infantry_counts)

        winner = get_winner(state, hex_index)
        send_back = []
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

        battle_events.append({
            "hex": list(state.grid.coord_of(hex_index)),
            "contributions_start": contributions_start,
            "archer_phase": full_log["archer_phase"],
            "rounds": full_log["rounds"],
            "winner": winner,
            "rectification": send_back,
        })

    return battle_events


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


# Labels for the 10 checkpoints a logged turn produces - same meaning as
# engine/turn.py's CHECKPOINT_LABELS, reused as-is by hex_visualizer.py.
CHECKPOINT_LABELS = ["Start", "Buy", "Move 1", "Move 2", "Move 3",
                      "Cav 1", "Cav 2", "Cav 3", "Cav 4", "Battle"]


def _snapshot_entry(state, hex_index, coord):
    """Per-hex log entry: {"q","r","s","city","troops","battle"} - same
    field shape as engine/turn.py's snapshot_hexes entries, so the
    existing visualizer code can read either engine's replay log without
    caring which one produced it."""
    entry = {
        "q": coord[0], "r": coord[1], "s": coord[2],
        "city": int(state.city_owner[hex_index]) if state.city_owner[hex_index] != NO_FACTION else None,
        "troops": None, "battle": None,
    }
    if state.locked[hex_index]:
        totals = faction_totals(state, hex_index)
        if totals:
            entry["battle"] = {
                "contributions": [
                    {"faction": f, "infantry": int(t[0]), "cavalry": int(t[1]), "archers": int(t[2])}
                    for f, t in totals.items()
                ]
            }
    elif state.army_faction[hex_index] != NO_FACTION:
        entry["troops"] = {
            "faction": int(state.army_faction[hex_index]),
            "infantry": int(state.army_units[hex_index, 0]),
            "cavalry": int(state.army_units[hex_index, 1]),
            "archers": int(state.army_units[hex_index, 2]),
            "frozen": bool(state.frozen[hex_index]),
        }
    return entry


def snapshot_hexes(state):
    """Full per-hex snapshot (all coords) - mirrors engine/turn.py's
    snapshot_hexes."""
    return [_snapshot_entry(state, i, coord) for i, coord in enumerate(state.grid.coords)]


def sparse_hexes(full_hexes):
    """Filters a snapshot down to only hexes with something on them -
    mirrors engine/turn.py's sparse_hexes."""
    return [e for e in full_hexes if e["city"] is not None or e["troops"] is not None or e["battle"] is not None]


def _player_stats_snapshot(state):
    return {
        f: {"silver": int(state.silver[f]), "kill_xp": int(state.kill_xp[f]), "alive": bool(state.alive[f])}
        for f in range(state.num_factions)
    }


def run_turn_and_log(state, decide_buy, decide_movement, decide_cavalry, decide_target, decide_rectification,
                      rng=None):
    """Same as run_turn, but also returns a turn_record capturing enough
    to replay/visualize the turn: a full board snapshot at every one of
    the turn's 10 checkpoints, the battle events, and player stats at
    each checkpoint.

    Deliberately simpler than engine/turn.py's run_turn_and_log: that
    version stores a sparse keyframe once per turn plus a diff per
    phase-step, reconstructed incrementally - built to keep replay file
    size proportional to how much actually happens, a real concern for
    large/long v1 games. This version just stores sparse_hexes(...)
    (only occupied/city/battle hexes, but computed fresh, not as a diff
    against the previous checkpoint) at every checkpoint independently -
    no incremental reconstruction, no risk of a diffing bug, and at the
    scale these games run at for now the size difference doesn't matter.
    Revisit if replay files ever get large enough for that to change.
    """
    rng = rng or random.Random()
    turn_number = state.turn_number

    checkpoints = [sparse_hexes(snapshot_hexes(state))]  # "Start" - before income

    apply_income_phase(state)
    player_stats = [_player_stats_snapshot(state)]  # matches v1: captured AFTER income

    buy_actions = {}
    for faction in range(state.num_factions):
        if not state.alive[faction]:
            continue
        legal = get_legal_buy_actions(state, faction)
        chosen = decide_buy[faction](state, faction, legal)
        if chosen:
            buy_actions[faction] = chosen
    apply_buy_phase(state, buy_actions)
    checkpoints.append(sparse_hexes(snapshot_hexes(state)))
    player_stats.append(_player_stats_snapshot(state))

    for step in range(MOVEMENT_STEPS):
        actions = {}
        for faction in range(state.num_factions):
            if not state.alive[faction]:
                continue
            legal = legal_movement_mask(state, faction)
            chosen = decide_movement[faction](state, faction, step, legal)
            if chosen:
                actions[faction] = chosen
        apply_movement_step(state, actions, cavalry_only=False)
        checkpoints.append(sparse_hexes(snapshot_hexes(state)))
        player_stats.append(_player_stats_snapshot(state))

    for step in range(CAVALRY_STEPS):
        actions = {}
        for faction in range(state.num_factions):
            if not state.alive[faction]:
                continue
            legal = legal_cavalry_mask(state, faction)
            chosen = decide_cavalry[faction](state, faction, step, legal)
            if chosen:
                actions[faction] = chosen
        apply_movement_step(state, actions, cavalry_only=True)
        checkpoints.append(sparse_hexes(snapshot_hexes(state)))
        player_stats.append(_player_stats_snapshot(state))

    battle_events = _run_battle_phase(state, decide_target, decide_rectification, rng)
    apply_terrain_effects(state)
    _update_elimination(state)
    checkpoints.append(sparse_hexes(snapshot_hexes(state)))
    player_stats.append(_player_stats_snapshot(state))

    state.turn_number += 1

    turn_record = {
        "turn_number": turn_number,
        "checkpoints": checkpoints,
        "battle_events": battle_events,
        "player_stats": player_stats,
    }
    return state, turn_record


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
