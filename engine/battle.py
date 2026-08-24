"""
Battle resolution for engine - ported from engine/battle.py.

See that module's docstring for the full rules rationale (archer phase,
per-round targeting/rolls/kills, cavalry dismounts, rectification). This
port is deliberately as literal as possible, including preserving the
exact ORDER every rng.randint(1, 20) call happens in, because that order
is what the parity tests key on: two independently-implemented engines
fed identically-seeded RNGs will only produce identical results if they
consume rolls in the same sequence. Where v1 iterates
`battle.contributions` (a list, in append order), v2 iterates battle
contribution slots 0..K-1 (in the same append order, by construction -
see movement.py's _start_or_extend_battle) - that equivalence is what
keeps faction iteration order (and therefore roll order, and therefore
results) matching.

target_fn(state, hex_index, faction) -> target_faction_or_None is the
external decision point (agent's job, same as v1's target_fn) - not
implemented here. rectify_overflow's `send_back` is likewise supplied by
the caller.

resolve_full_battle returns a full structured log ({"archer_phase": [...],
"rounds": [...]}), same shape as engine/battle.py's - death/dismount
entries are plain dicts with string unit types ("infantry"/"cavalry"/
"archers"), not the leaner int-indexed tuples this module uses
internally elsewhere, specifically so this log can be handed straight to
the existing replay/visualization code (hex_visualizer.py's
compute_battle_table) unchanged - see engine/turn.py's
run_turn_and_log.
"""

import numpy as np

from .state import (
    MAX_STACK_SIZE,
    NO_FACTION,
    NO_ORIGIN,
    SPAWN_CAPS,
    UNIT_TYPES,
    count_units_in_play,
)

DEATH_PRIORITY = (0, 1, 2)  # infantry, cavalry, archers
MAX_ROUNDS_SAFETY_CAP = 50  # pure infinite-loop guard - real battles resolve in a handful of rounds
CAPITAL_DEFENSE_SHOTS = 2
OUTPOST_DEFENSE_SHOTS = 1


def _battle_faction_order(state, hex_index):
    """Faction ids present in this hex's battle contributions, in
    first-appearance (slot) order - mirrors engine/battle.py's
    battle.factions() (which de-dupes battle.contributions, a list, by
    first appearance)."""
    order = []
    seen = set()
    for k in range(state.battle_faction.shape[1]):
        f = int(state.battle_faction[hex_index, k])
        if f == NO_FACTION or f in seen:
            continue
        seen.add(f)
        order.append(f)
    return order


def faction_totals(state, hex_index):
    """{faction: int[3]} in first-appearance order - mirrors
    engine/battle.py's battle.faction_totals()."""
    totals = {}
    for f in _battle_faction_order(state, hex_index):
        mask = state.battle_faction[hex_index] == f
        totals[f] = state.battle_units[hex_index][mask].sum(axis=0)
    return totals


def faction_alive_totals(state, hex_index):
    """{faction: total_units} for factions still alive in this battle -
    mirrors engine/battle.py's faction_alive_totals()."""
    return {f: t for f, t in faction_totals(state, hex_index).items() if int(t.sum()) > 0}


def get_legal_target_actions(state, hex_index, faction):
    """Valid targets for `faction` this round: any other faction still
    alive in the battle - mirrors engine/turn.py's
    get_legal_target_actions."""
    totals = faction_totals(state, hex_index)
    return [f for f in _battle_faction_order(state, hex_index) if f != faction and int(totals[f].sum()) > 0]


def _kills_for_roll(roll, attacker_total_units):
    if roll <= 5:
        return 0
    if roll <= 15:
        return 1
    return 1 if attacker_total_units == 1 else 2


def _apply_kills_to_faction(state, hex_index, target_faction, num_kills, killer_faction, death_log):
    """Removes up to num_kills units from target_faction's presence in
    this battle (infantry -> cavalry -> archers cascade, earliest
    contribution slots first), appending one {"faction", "unit_type",
    "count", "killer"} dict per removal - mirrors engine/battle.py's
    _apply_kills_to_faction (dict shape and all, see module docstring)."""
    remaining = num_kills
    for ut in DEATH_PRIORITY:
        if remaining <= 0:
            break
        for k in range(state.battle_faction.shape[1]):
            if remaining <= 0:
                break
            if state.battle_faction[hex_index, k] != target_faction:
                continue
            take = min(int(state.battle_units[hex_index, k, ut]), remaining)
            if take > 0:
                state.battle_units[hex_index, k, ut] -= take
                remaining -= take
                death_log.append(
                    {"faction": target_faction, "unit_type": UNIT_TYPES[ut], "count": take, "killer": killer_faction}
                )


def apply_archer_abilities(state, hex_index, rng):
    """Runs once, before round 1. Returns a death_log list of
    {"faction", "unit_type", "count", "killer"} dicts - mirrors
    engine/battle.py's apply_archer_abilities."""
    order = _battle_faction_order(state, hex_index)
    totals = faction_totals(state, hex_index)
    alive = {f: t for f, t in totals.items() if int(t.sum()) > 0}
    death_log = []

    for faction in order:
        archers = int(totals[faction][2])
        if archers <= 0 or faction not in alive:
            continue
        rivals = {f: int(alive[f].sum()) for f in order if f != faction and f in alive}
        if not rivals:
            continue
        target = max(rivals, key=lambda f: rivals[f])

        kills = 0
        for _ in range(archers):
            if rng.randint(1, 20) >= 11:
                kills += 1
        if kills > 0:
            _apply_kills_to_faction(state, hex_index, target, kills, faction, death_log)

    return death_log


def apply_structure_defense_shots(state, hex_index, rng):
    """Runs once, before round 1 (and before real Archer units get their
    own ability - see apply_archer_abilities). A capital or outpost gets
    free defensive shots against whoever's attacking its tile, even if
    its owner has no units there to defend with: 2 shots for a capital,
    1 for an outpost, each 11-20 = 1 kill against the largest attacking
    army, same math as the real Archers ability but a deliberately
    separate mechanic so it can be tuned independently later. Returns a
    death_log list in the same shape as apply_archer_abilities."""
    owner = int(state.city_owner[hex_index])
    if owner == NO_FACTION:
        return []

    totals = faction_totals(state, hex_index)
    alive = {f: t for f, t in totals.items() if int(t.sum()) > 0}
    rivals = {f: int(t.sum()) for f, t in alive.items() if f != owner}
    if not rivals:
        return []
    target = max(rivals, key=lambda f: rivals[f])

    shots = CAPITAL_DEFENSE_SHOTS if state.is_capital[hex_index] else OUTPOST_DEFENSE_SHOTS
    kills = 0
    for _ in range(shots):
        if rng.randint(1, 20) >= 11:
            kills += 1

    death_log = []
    if kills > 0:
        _apply_kills_to_faction(state, hex_index, target, kills, owner, death_log)
    return death_log


def _resolve_targets(state, hex_index, target_choices):
    """target_choices: {faction: target_or_None}, in the caller's
    iteration order. Returns {attacker: target} for attacks actually
    allowed this round (conflict rule: highest total units wins a
    contested target) - mirrors engine/battle.py's _resolve_targets."""
    totals = faction_totals(state, hex_index)
    unit_counts = {f: int(t.sum()) for f, t in totals.items()}

    by_target = {}
    for attacker, target in target_choices.items():
        if target is None:
            continue
        by_target.setdefault(target, []).append(attacker)

    resolved = {}
    for target, attackers in by_target.items():
        if len(attackers) == 1:
            resolved[attackers[0]] = target
        else:
            winner = max(attackers, key=lambda f: (unit_counts.get(f, 0), -f))
            resolved[winner] = target
    return resolved


def resolve_round(state, hex_index, target_choices, rng, infantry_counts):
    """Runs one full round in place: targeting conflicts, rolls,
    simultaneous kill application, then cavalry dismounts. Returns a
    dict matching engine/battle.py's resolve_round's round_log shape
    (target_choices_submitted, resolved_targets, rolls, kills_dealt,
    deaths, dismounts) - "deaths" also drives kill-XP crediting.

    infantry_counts: {faction: current_infantry_in_play}, a running
    tally the caller maintains (shared across every battle resolved in
    the same turn, same as v1) so dismount cap checks don't need a fresh
    board scan on every roll.
    """
    resolved_targets = _resolve_targets(state, hex_index, target_choices)
    totals = faction_totals(state, hex_index)
    unit_counts = {f: int(t.sum()) for f, t in totals.items()}

    rolls = {}
    kills_dealt = {}
    pending_kills = []
    for attacker, target in resolved_targets.items():
        attacker_units = unit_counts.get(attacker, 0)
        if attacker_units <= 0:
            continue
        roll = rng.randint(1, 20)
        kills = _kills_for_roll(roll, attacker_units)
        rolls[attacker] = roll
        kills_dealt[attacker] = kills
        if kills > 0:
            pending_kills.append((target, kills, attacker))

    death_log = []
    cav_died_by_faction = {}
    for target, kills, killer in pending_kills:
        cav_before = int(state.battle_units[hex_index, state.battle_faction[hex_index] == target, 1].sum())
        _apply_kills_to_faction(state, hex_index, target, kills, killer, death_log)
        cav_after = int(state.battle_units[hex_index, state.battle_faction[hex_index] == target, 1].sum())
        died = cav_before - cav_after
        if died > 0:
            cav_died_by_faction[target] = cav_died_by_faction.get(target, 0) + died

    dismount_log = []
    for faction, died_count in cav_died_by_faction.items():
        for _ in range(died_count):
            if rng.randint(1, 20) < 14:
                dismount_log.append({"faction": faction, "success": False})
                continue
            if infantry_counts.get(faction, 0) >= int(SPAWN_CAPS[0]):
                dismount_log.append({"faction": faction, "success": False, "reason": "cap"})
                continue
            for k in range(state.battle_faction.shape[1]):
                if state.battle_faction[hex_index, k] == faction:
                    state.battle_units[hex_index, k, 0] += 1
                    break
            infantry_counts[faction] = infantry_counts.get(faction, 0) + 1
            dismount_log.append({"faction": faction, "success": True})

    state.battle_round[hex_index] += 1
    return {
        "target_choices_submitted": dict(target_choices),
        "resolved_targets": dict(resolved_targets),
        "rolls": rolls,
        "kills_dealt": kills_dealt,
        "deaths": death_log,
        "dismounts": dismount_log,
    }


def is_battle_over(state, hex_index):
    return len(faction_alive_totals(state, hex_index)) <= 1


def get_winner(state, hex_index):
    alive = faction_alive_totals(state, hex_index)
    if len(alive) == 1:
        return next(iter(alive))
    return None


def resolve_full_battle(state, hex_index, target_fn, rng, infantry_counts=None):
    """Runs the whole battle to completion, in place. `target_fn(state,
    hex_index, faction) -> target_faction_or_None` is called once per
    living faction per round (the agent decision point).
    `infantry_counts`, if not provided, is built from a fresh scan of
    just this battle's factions (fine for isolated calls/tests; the
    real turn orchestration should share one tally across every battle
    resolved that turn - see engine/turn.py's _run_battle_phase for why).

    Returns {"structure_phase": [...], "archer_phase": [...], "rounds":
    [round_log, ...]} - "rounds" and "archer_phase" match the shape of
    engine/battle.py's resolve_full_battle, for replay/visualization (see
    module docstring); "structure_phase" is new (see
    apply_structure_defense_shots) and not yet consumed by any replay
    viewer. Always computed, same as v1 - cheap, bounded by actual rounds
    fought - even if a given caller (e.g. run_turn, as opposed to
    run_turn_and_log) doesn't keep it.
    """
    if infantry_counts is None:
        infantry_counts = {f: count_units_in_play(state, f, 0) for f in _battle_faction_order(state, hex_index)}

    players_kill_xp = state.kill_xp

    structure_phase = apply_structure_defense_shots(state, hex_index, rng)
    for entry in structure_phase:
        players_kill_xp[entry["killer"]] += entry["count"]

    archer_phase = apply_archer_abilities(state, hex_index, rng)
    for entry in archer_phase:
        players_kill_xp[entry["killer"]] += entry["count"]

    rounds = []
    rounds_run = 0
    while not is_battle_over(state, hex_index) and rounds_run < MAX_ROUNDS_SAFETY_CAP:
        order = _battle_faction_order(state, hex_index)
        totals = faction_totals(state, hex_index)
        target_choices = {}
        for faction in order:
            if int(totals.get(faction, np.zeros(3)).sum()) <= 0:
                continue
            target_choices[faction] = target_fn(state, hex_index, faction)

        round_result = resolve_round(state, hex_index, target_choices, rng, infantry_counts)
        rounds.append(round_result)
        for entry in round_result["deaths"]:
            players_kill_xp[entry["killer"]] += entry["count"]

        rounds_run += 1

    return {"structure_phase": structure_phase, "archer_phase": archer_phase, "rounds": rounds}


def rectify_overflow(state, hex_index, winner_faction, send_back, cap=MAX_STACK_SIZE):
    """After a battle resolves, if the winning stack exceeds `cap` units,
    the winner sends the excess back to their own contributing origin
    hexes. `send_back`: list of {"origin_hex": hex_index_or_None,
    "units": int[3]}. Units whose origin_hex is None/invalid, or that
    `send_back` doesn't account for, are trimmed off (infantry -> cavalry
    -> archers, same cascade used everywhere else) rather than left
    sitting above `cap` - mirrors engine/battle.py's rectify_overflow,
    generalized with a `cap` parameter so turn.py can force a full
    eviction (cap=0) from a capital a foreign faction just won a battle
    on (see turn.py's _run_battle_phase - capitals are uncapturable, so
    an attacker who wins there is never allowed to actually occupy it).

    City ownership is NOT touched here: neither a capital (uncapturable)
    nor an outpost (destroyed rather than captured - see turn.py) ever
    changes hands by occupation anymore, so that's turn.py's job, not
    this generic stack-trimming function's."""
    winning_units = faction_totals(state, hex_index)[winner_faction].copy()

    for entry in send_back:
        origin = entry["origin_hex"]
        units = entry["units"]
        valid_origin = origin is not None and 0 <= origin < state.num_hexes
        for ut in range(3):
            take = min(int(units[ut]), int(winning_units[ut]))
            winning_units[ut] -= take
            if valid_origin:
                if state.army_faction[origin] == NO_FACTION:
                    state.army_faction[origin] = winner_faction
                if state.army_faction[origin] == winner_faction:
                    state.army_units[origin, ut] += take
            # else: those units are simply lost

    total_remaining = int(winning_units.sum())
    if total_remaining > cap:
        excess = total_remaining - cap
        for ut in DEATH_PRIORITY:
            take = min(int(winning_units[ut]), excess)
            winning_units[ut] -= take
            excess -= take
            if excess <= 0:
                break

    if int(winning_units.sum()) > 0:
        state.army_faction[hex_index] = winner_faction
        state.army_units[hex_index] = winning_units
    else:
        state.army_faction[hex_index] = NO_FACTION
        state.army_units[hex_index] = 0
    state.frozen[hex_index] = False
    state.locked[hex_index] = False

    state.battle_faction[hex_index] = NO_FACTION
    state.battle_origin[hex_index] = NO_ORIGIN
    state.battle_units[hex_index] = 0
    state.battle_round[hex_index] = 0
    state.battle_order.remove(hex_index)

    return state
