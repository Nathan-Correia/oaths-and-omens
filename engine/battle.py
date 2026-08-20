"""
Battle resolution.

Sequence for one battle:
  1. Archer abilities fire once, before round 1 (see simplification
     note below).
  2. Rounds repeat until only one faction (or zero) has units left:
       a. every faction with units simultaneously picks a target
          faction (get_legal_target_actions / apply via target_choices)
       b. target conflicts resolved: if 2+ factions target the same
          faction, only the one with more total units in the battle
          actually attacks this round - the other(s) don't attack.
       c. one d20 roll per faction-with-a-valid-target, plus terrain
          bonuses (see below); kill table: 1-5 -> 0, 6-15 -> 1, 16-20
          -> 2 (except: if the ATTACKER has exactly 1 unit left, 16-20
          -> 1 kill instead of 2 - this reflects "if you have 1 unit
          left, 16-20 is a single kill").
       d. kills applied simultaneously (mutual kills both happen).
          Death priority per faction: infantry -> cavalry -> archers,
          cascading past empty types.
       e. cavalry death ability rolls for each cavalry unit that died
          this round: 14-20 -> a dismounted infantry unit joins that
          faction's forces in this same battle immediately (subject to
          their concurrent unit cap), before the next round's
          is-the-battle-over check - so a last-cavalry dismount can
          keep a faction in the fight.
       f. attacker of each kill gains kill-XP (a plain currency count)
          for the units that died.
  3. When one faction remains, that faction wins the hex. If the
     winning stack exceeds 6 units, the winner chooses how to send the
     overflow back to any of their own contributing origin hexes
     (rectify_overflow) - if none are valid, those units are lost.

Terrain bonuses (+2 to a d20 roll each, stacking, capped at 20):
  - Forest: a faction gets +2 on every roll (including the archer
    ability roll) if any of its contributions to this battle
    originated from a forest hex - "attacking out of a forest."
  - City: a faction gets +2 on every roll if it owns the city sitting
    on the battle's hex - defending their own city. Ownership can't
    change mid-battle (only rectify_overflow transfers it, once the
    fight is over), so this is safe to check at any point.

SIMPLIFICATION: the archer pre-round ability targets whichever enemy
faction currently has the most units in the battle, rather than
reusing the round-1 target choice. This decouples the ability from the
per-round targeting decision point and keeps the engine simpler; worth
revisiting if archer balance ends up sensitive to who they hit first.
"""

import random

from .state import UNIT_TYPES, MAX_STACK_SIZE, SPAWN_CAPS, count_units_in_play

DEATH_PRIORITY = ["infantry", "cavalry", "archers"]
MAX_ROUNDS_SAFETY_CAP = 200
TERRAIN_BONUS = 2


def _roll_d20(rng):
    return rng.randint(1, 20)


def _terrain_bonus(state, battle, faction):
    """+2 if any of this faction's contributions originated from a
    forest hex ("attacking out of a forest"), +2 if this faction owns
    the city on the battle's hex (defending their own city) - these
    stack, per the rulebook. City ownership is looked up on the live
    board rather than snapshotted, but it can't change mid-battle
    (only rectify_overflow transfers it, once the fight is over), so
    this is safe to check at any point during resolution."""
    bonus = 0
    for c in battle.contributions:
        if c["faction"] == faction:
            origin = c.get("origin_hex")
            if origin is not None:
                oh = state.board.get(tuple(origin))
                if oh and oh.terrain == "forest":
                    bonus += TERRAIN_BONUS
                    break

    h = state.board.get(battle.hex_coord)
    if h and h.city_owner == faction:
        bonus += TERRAIN_BONUS

    return bonus


def _kills_for_roll(roll, attacker_total_units):
    if roll <= 5:
        return 0
    if roll <= 15:
        return 1
    # 16-20
    return 1 if attacker_total_units == 1 else 2


def faction_alive_totals(battle):
    """{faction: total_units_remaining} for factions still in the fight."""
    totals = battle.faction_totals()
    return {f: sum(t.values()) for f, t in totals.items() if sum(t.values()) > 0}


def apply_archer_abilities(battle, rng, state):
    """Runs once, before round 1. Returns a log list of kills applied.
    Terrain bonuses apply to the archers' ability roll the same as any
    other battle roll."""
    totals = battle.faction_totals()
    alive = faction_alive_totals(battle)
    log = []

    for faction, comp in totals.items():
        archers = comp["archers"]
        if archers <= 0 or faction not in alive:
            continue
        rivals = {f: n for f, n in alive.items() if f != faction}
        if not rivals:
            continue
        target = max(rivals, key=lambda f: rivals[f])

        bonus = _terrain_bonus(state, battle, faction)
        kills = 0
        for _ in range(archers):
            if min(20, _roll_d20(rng) + bonus) >= 11:
                kills += 1
        if kills > 0:
            _apply_kills_to_faction(battle, target, kills, killer_faction=faction, log=log)

    return log


def _resolve_targets(battle, target_choices):
    """target_choices: {faction: target_faction_or_None}. Returns the
    subset of choices that are actually allowed to attack this round
    (conflicts resolved: highest total units wins the right to attack
    a contested target; others in the conflict don't attack)."""
    totals = battle.faction_totals()
    unit_counts = {f: sum(t.values()) for f, t in totals.items()}

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


def _apply_kills_to_faction(battle, faction, num_kills, killer_faction, log):
    """Removes up to num_kills units from `faction`'s total presence in
    the battle (infantry -> cavalry -> archers cascade), spread across
    that faction's contributions (earliest contributions first), and
    credits killer_faction with kill-XP tokens for each unit removed.
    Also returns/logs cavalry deaths for the death-ability roll."""
    remaining = num_kills
    for unit_type in DEATH_PRIORITY:
        if remaining <= 0:
            break
        for contribution in battle.contributions:
            if contribution["faction"] != faction:
                continue
            take = min(contribution[unit_type], remaining)
            if take > 0:
                contribution[unit_type] -= take
                remaining -= take
                log.append({"faction": faction, "unit_type": unit_type, "count": take, "killer": killer_faction})
            if remaining <= 0:
                break


def resolve_round(battle, target_choices, state, rng):
    """Runs one full round: targeting conflicts, rolls, simultaneous
    kill application, then cavalry dismounts. Returns a round_log dict:
      "target_choices_submitted": {faction: target_or_None} - what each
        faction actually asked for, before conflict resolution.
      "resolved_targets": {faction: target} - who actually got to
        attack this round, after the higher-unit-count conflict rule.
      "rolls": {faction: raw_d20_roll} - only for factions that attacked.
      "kills_dealt": {faction: num_kills} - resulting kill count per roll.
      "deaths": [{"faction","unit_type","count","killer"}, ...] real
        unit deaths (used by the caller to credit kill-XP).
      "dismounts": [{"faction","success","reason"?}, ...] one entry per
        cavalry unit that died this round.

    Dismounted infantry are added directly into one of the dying
    faction's existing battle contributions - they become live
    combatants immediately, before the next round's is_battle_over
    check, so a last-cavalry dismount can keep a faction in the fight.
    """
    resolved_targets = _resolve_targets(battle, target_choices)
    totals = battle.faction_totals()
    unit_counts = {f: sum(t.values()) for f, t in totals.items()}

    # roll first, then apply all kills simultaneously (mutual kills both happen)
    rolls = {}
    kills_dealt = {}
    pending_kills = []  # (target_faction, num_kills, killer_faction)
    for attacker, target in resolved_targets.items():
        attacker_units = unit_counts.get(attacker, 0)
        if attacker_units <= 0:
            continue
        bonus = _terrain_bonus(state, battle, attacker)
        roll = min(20, _roll_d20(rng) + bonus)
        kills = _kills_for_roll(roll, attacker_units)
        rolls[attacker] = roll
        kills_dealt[attacker] = kills
        if kills > 0:
            pending_kills.append((target, kills, attacker))

    death_log = []
    cav_died_by_faction = {}
    for target, kills, killer in pending_kills:
        # snapshot cavalry count before applying kills, to know how many cavalry died
        cav_before = sum(c["cavalry"] for c in battle.contributions if c["faction"] == target)
        _apply_kills_to_faction(battle, target, kills, killer, death_log)
        cav_after = sum(c["cavalry"] for c in battle.contributions if c["faction"] == target)
        died = cav_before - cav_after
        if died > 0:
            cav_died_by_faction[target] = cav_died_by_faction.get(target, 0) + died

    # cavalry death ability: one roll per cavalry unit that died this round
    dismount_log = []
    for faction, died_count in cav_died_by_faction.items():
        for _ in range(died_count):
            if _roll_d20(rng) < 14:
                dismount_log.append({"faction": faction, "success": False})
                continue
            if count_units_in_play(state, faction, "infantry") >= SPAWN_CAPS["infantry"]:
                dismount_log.append({"faction": faction, "success": False, "reason": "cap"})
                continue
            for c in battle.contributions:
                if c["faction"] == faction:
                    c["infantry"] += 1
                    break
            dismount_log.append({"faction": faction, "success": True})

    battle.round_number += 1
    return {
        "target_choices_submitted": dict(target_choices),
        "resolved_targets": dict(resolved_targets),
        "rolls": rolls,
        "kills_dealt": kills_dealt,
        "deaths": death_log,
        "dismounts": dismount_log,
    }


def is_battle_over(battle):
    return len(faction_alive_totals(battle)) <= 1


def get_winner(battle):
    alive = faction_alive_totals(battle)
    if len(alive) == 1:
        return next(iter(alive))
    return None  # mutual annihilation, or battle not over yet


def resolve_full_battle(battle, target_fn, state, rng=None):
    """Runs the whole battle to completion.

    target_fn(battle, faction) -> target_faction_or_None, called once
    per living faction per round (this is the agent decision point).
    state: the full GameState - needed (not just `battle`) because the
    cavalry dismount ability has to check the faction's concurrent
    unit cap across the whole board, not just this battle.

    Returns a full log: {"archer_phase": [...], "rounds": [round_log, ...]}
    - full per-round detail (targets, rolls, deaths, dismounts), for
    replay/debugging and the future step-by-step battle viewer.
    """
    rng = rng or random.Random()
    players = state.players
    full_log = {"archer_phase": [], "rounds": []}

    archer_log = apply_archer_abilities(battle, rng, state)
    full_log["archer_phase"] = archer_log
    for entry in archer_log:
        if entry["killer"] in players:
            players[entry["killer"]].kill_xp += entry["count"]

    rounds_run = 0
    while not is_battle_over(battle) and rounds_run < MAX_ROUNDS_SAFETY_CAP:
        target_choices = {}
        for faction in battle.factions():
            totals = battle.faction_totals()
            if sum(totals.get(faction, {}).values()) <= 0:
                continue
            target_choices[faction] = target_fn(battle, faction)

        round_log = resolve_round(battle, target_choices, state, rng)
        full_log["rounds"].append(round_log)

        for entry in round_log["deaths"]:
            killer = entry["killer"]
            if killer in players:
                players[killer].kill_xp += entry["count"]

        rounds_run += 1

    return full_log


def rectify_overflow(state, battle_hex, winner_faction, send_back):
    """After a battle resolves, if the winning stack exceeds 6 units,
    the winner sends overflow back to their own contributing origin
    hexes. `send_back`: list of {"origin_hex": coord, "units": {...}}
    chosen by the winner (or their agent). Units whose origin_hex is
    no longer a valid destination (off board / impassable) are lost."""
    battle = state.battles.get(battle_hex)
    if battle is None:
        return state

    totals = battle.faction_totals()[winner_faction]
    winning_army = {"faction": winner_faction, "frozen": False, **totals}

    for entry in send_back:
        origin = tuple(entry["origin_hex"])
        units = entry["units"]
        oh = state.board.get(origin)
        for ut in UNIT_TYPES:
            take = min(units.get(ut, 0), winning_army[ut])
            winning_army[ut] -= take
            if oh is not None:
                if oh.army is None:
                    oh.army = {"faction": winner_faction, "infantry": 0, "cavalry": 0, "archers": 0, "frozen": False}
                if oh.army["faction"] == winner_faction:
                    oh.army[ut] += take
            # if oh is None / invalid, those units are simply lost

    # clamp whatever's left at the battle hex to the stack cap as a final safety net
    total_remaining = sum(winning_army[ut] for ut in UNIT_TYPES)
    if total_remaining > MAX_STACK_SIZE:
        winning_army["infantry"] = max(0, winning_army["infantry"] - (total_remaining - MAX_STACK_SIZE))

    h = state.board[battle_hex]
    h.army = winning_army if sum(winning_army[ut] for ut in UNIT_TYPES) > 0 else None
    h.locked = False
    if h.city_owner is not None and h.army is not None:
        h.city_owner = winner_faction
    del state.battles[battle_hex]
    return state