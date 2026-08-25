"""
Turn orchestration for engine - ported from engine/turn.py's run_turn,
plus a run_turn_and_log for replay/visualization (see this module's
run_turn_and_log docstring for how its log format differs from v1's).

Agents aren't a formal class hierarchy here (unlike v1's BaseAgent) - each
decision point takes a plain callback instead, looked up per-faction from
a {faction: callable} dict (mirroring v1's {faction: agent}). Deliberate:
the real "agent" for engine will eventually be a neural policy with
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
  decide_rectification(state, hex_index, winner_faction, cap) -> [{"origin_hex", "units"}, ...]
    `cap` is normally MAX_STACK_SIZE (send back only the overflow above 6), but is 0 when
    the winner just won a battle on a foreign capital - capitals are uncapturable, so that
    winner has to send EVERYTHING back (see _run_battle_phase).
"""

import random

import numpy as np

from .battle import faction_totals, get_winner, rectify_overflow, resolve_full_battle
from .buy import apply_buy_phase, get_legal_buy_actions
from .income import apply_income_phase
from .movement import apply_movement_step, legal_cavalry_mask, legal_movement_mask
from .state import MAX_STACK_SIZE, NO_FACTION, NO_ORIGIN, count_units_in_play
from .terrain import apply_terrain_effects

MOVEMENT_STEPS = 3
CAVALRY_STEPS = 4
VP_TO_WIN = 50
OUTPOST_VP_PER_ROUND = 1
OUTPOST_DESTROY_VP = 2


def _run_battle_phase(state, decide_target, decide_rectification, rng):
    """Resolves every pending battle, in state.battle_order (battle
    creation order - see state.py's module docstring for why this has to
    match v1's dict-insertion-order semantics rather than e.g. hex-index
    order: the dismount infantry cap tally below is shared across every
    battle resolved this turn, so processing order can affect outcomes
    near the cap). Returns a list of per-battle event logs, same shape
    as engine/turn.py's _run_battle_phase - always built (cheap, bounded
    by rounds actually fought), even though only run_turn_and_log ends
    up keeping it.

    RULE CHANGE - capitals/outposts: neither is capturable by occupation
    anymore (see movement.py). When the winner of a battle on a hex isn't
    that hex's city_owner:
      - a capital evicts the winner entirely (cap=0 rectification, no
        ownership change - "you can't stand units in another player's
        capital").
      - an outpost is destroyed (city_owner cleared) and the winner gets
        OUTPOST_DESTROY_VP; the winner keeps standing there (normal
        cap=MAX_STACK_SIZE rectification), same as any other battle hex.
    """
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
            owner = int(state.city_owner[hex_index])
            cap = MAX_STACK_SIZE
            if owner != NO_FACTION and owner != winner:
                if state.is_capital[hex_index]:
                    cap = 0
                else:
                    state.city_owner[hex_index] = NO_FACTION
                    state.victory_points[winner] += OUTPOST_DESTROY_VP

            send_back = decide_rectification[winner](state, hex_index, winner, cap)
            rectify_overflow(state, hex_index, winner, send_back, cap=cap)

        battle_events.append({
            "hex": list(state.grid.coord_of(hex_index)),
            "contributions_start": contributions_start,
            "structure_phase": full_log["structure_phase"],
            "archer_phase": full_log["archer_phase"],
            "rounds": full_log["rounds"],
            "winner": winner,
            "rectification": send_back,
        })

    return battle_events


def apply_victory_points(state):
    """End-of-round VP tally (the win condition - see get_game_winner):
    1 point per outpost currently controlled, capitals don't count.
    Destroying an enemy outpost is awarded separately, immediately, in
    _run_battle_phase (OUTPOST_DESTROY_VP) - this only covers the
    recurring per-round income."""
    for faction in range(state.num_factions):
        outposts = int(np.sum((state.city_owner == faction) & ~state.is_capital))
        state.victory_points[faction] += outposts * OUTPOST_VP_PER_ROUND


def get_game_winner(state):
    """None until some faction's VP total has reached VP_TO_WIN. Among
    every faction at or above VP_TO_WIN, the strict highest total wins
    outright; an exact tie for the top total is broken by whoever placed
    their capital later (see rulebook's Win Condition), using
    state.capital_settle_order - set once per faction by
    placement.py's run_city_setup, a single incrementing counter, so no
    two factions can ever tie on it and no further randomness is needed
    here."""
    top = int(np.max(state.victory_points))
    if top < VP_TO_WIN:
        return None
    contenders = [f for f in range(state.num_factions) if int(state.victory_points[f]) == top]
    if len(contenders) == 1:
        return contenders[0]
    return max(contenders, key=lambda f: int(state.capital_settle_order[f]))


def run_turn(state, decide_buy, decide_movement, decide_cavalry, decide_target, decide_rectification, rng=None):
    """Mutates and returns `state`. Each decide_* argument is a
    {faction: callable} dict - see module docstring for each callable's
    signature. Elimination doesn't exist anymore (capitals are
    uncapturable - see movement.py/_run_battle_phase), so every faction
    is asked for a decision every turn."""
    rng = rng or random.Random()

    apply_income_phase(state)

    buy_actions = {}
    for faction in range(state.num_factions):
        legal = get_legal_buy_actions(state, faction)
        buy_actions[faction] = decide_buy[faction](state, faction, legal)
    apply_buy_phase(state, buy_actions)

    for step in range(MOVEMENT_STEPS):
        actions = {}
        for faction in range(state.num_factions):
            legal = legal_movement_mask(state, faction)
            actions[faction] = decide_movement[faction](state, faction, step, legal)
        apply_movement_step(state, actions, cavalry_only=False)

    for step in range(CAVALRY_STEPS):
        actions = {}
        for faction in range(state.num_factions):
            legal = legal_cavalry_mask(state, faction)
            actions[faction] = decide_cavalry[faction](state, faction, step, legal)
        apply_movement_step(state, actions, cavalry_only=True)

    _run_battle_phase(state, decide_target, decide_rectification, rng)
    apply_terrain_effects(state)
    apply_victory_points(state)

    state.turn_number += 1
    return state


# Labels for the 10 checkpoints a logged turn produces - same meaning as
# engine/turn.py's CHECKPOINT_LABELS, reused as-is by hex_visualizer.py.
CHECKPOINT_LABELS = ["Start", "Buy", "Move 1", "Move 2", "Move 3",
                      "Cav 1", "Cav 2", "Cav 3", "Cav 4", "Battle"]


def _snapshot_entry(state, hex_index, coord):
    """Per-hex log entry: {"q","r","s","city","troops","battle"}. "city"
    is None, or {"faction", "is_capital"} - capitals and outposts share
    city_owner but render as different icons (hex_visualizer.py's
    draw_city_icon vs. draw_outpost_icon), so is_capital has to travel
    with the log entry rather than being re-derived some other way."""
    if state.city_owner[hex_index] != NO_FACTION:
        city = {"faction": int(state.city_owner[hex_index]), "is_capital": bool(state.is_capital[hex_index])}
    else:
        city = None
    entry = {
        "q": coord[0], "r": coord[1], "s": coord[2],
        "city": city,
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
        f: {
            "silver": int(state.silver[f]),
            "kill_xp": int(state.kill_xp[f]),
            "victory_points": int(state.victory_points[f]),
            "alive": bool(state.alive[f]),
        }
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
            legal = legal_cavalry_mask(state, faction)
            chosen = decide_cavalry[faction](state, faction, step, legal)
            if chosen:
                actions[faction] = chosen
        apply_movement_step(state, actions, cavalry_only=True)
        checkpoints.append(sparse_hexes(snapshot_hexes(state)))
        player_stats.append(_player_stats_snapshot(state))

    battle_events = _run_battle_phase(state, decide_target, decide_rectification, rng)
    apply_terrain_effects(state)
    apply_victory_points(state)
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


def check_game_end(state, max_turns=None):
    """True once the game should stop: the VP win condition has been hit
    (see get_game_winner) - this is a rule, checked unconditionally. If
    `max_turns` is given, it's purely an infra safety net against a
    runaway/no-outposts-ever-built game, NOT part of the rules (there is
    no turn timer anymore); omit it to let the game run until someone
    actually wins."""
    if get_game_winner(state) is not None:
        return True
    return max_turns is not None and state.turn_number >= max_turns


def tally_final_score(state):
    """The win condition IS the score now - just victory_points per
    faction (see get_game_winner/apply_victory_points). Kept as its own
    function for API parity with callers that want a final {faction:
    score} dict."""
    return {f: int(state.victory_points[f]) for f in range(state.num_factions)}
