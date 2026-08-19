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
       c. one d20 roll per faction-with-a-valid-target; kill table:
          1-5 -> 0, 6-15 -> 1, 16-20 -> 2 (except: if the ATTACKER has
          exactly 1 unit left, 16-20 -> 1 kill instead of 2 - this
          reflects "if you have 1 unit left, 16-20 is a single kill").
       d. kills applied simultaneously (mutual kills both happen).
          Death priority per faction: infantry -> cavalry -> archers,
          cascading past empty types.
       e. cavalry death ability rolls at the end of the round for each
          cavalry unit that died this round: 14-20 -> owner queues a
          free infantry (resolved automatically next buy phase).
       f. attacker of each kill gains kill_xp tokens for the units
          that died.
  3. When one faction remains, that faction wins the hex. If the
     winning stack exceeds 6 units, the winner chooses how to send the
     overflow back to any of their own contributing origin hexes
     (rectify_overflow) - if none are valid, those units are lost.

SIMPLIFICATION: the archer pre-round ability targets whichever enemy
faction currently has the most units in the battle, rather than
reusing the round-1 target choice. This decouples the ability from the
per-round targeting decision point and keeps the engine simpler; worth
revisiting if archer balance ends up sensitive to who they hit first.
"""

import random

from .state import UNIT_TYPES, MAX_STACK_SIZE

DEATH_PRIORITY = ["infantry", "cavalry", "archers"]
MAX_ROUNDS_SAFETY_CAP = 200


def _roll_d20(rng):
    return rng.randint(1, 20)


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


def apply_archer_abilities(battle, rng):
    """Runs once, before round 1. Returns a log list of kills applied."""
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

        kills = 0
        for _ in range(archers):
            if _roll_d20(rng) >= 11:
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


def resolve_round(battle, target_choices, rng):
    """Runs one full round: targeting conflicts, rolls, simultaneous
    kill application, cavalry death-ability rolls. Returns a log list
    (each entry: {"faction","unit_type","count","killer"}) plus a list
    of cavalry-death-ability results: [{"faction": f, "free_infantry": bool}]."""
    resolved_targets = _resolve_targets(battle, target_choices)
    totals = battle.faction_totals()
    unit_counts = {f: sum(t.values()) for f, t in totals.items()}

    # roll first, then apply all kills simultaneously (mutual kills both happen)
    pending_kills = []  # (target_faction, num_kills, killer_faction)
    for attacker, target in resolved_targets.items():
        attacker_units = unit_counts.get(attacker, 0)
        if attacker_units <= 0:
            continue
        roll = _roll_d20(rng)
        kills = _kills_for_roll(roll, attacker_units)
        if kills > 0:
            pending_kills.append((target, kills, attacker))

    death_log = []
    for target, kills, killer in pending_kills:
        # snapshot cavalry count before applying kills, to know how many cavalry died
        cav_before = sum(c["cavalry"] for c in battle.contributions if c["faction"] == target)
        _apply_kills_to_faction(battle, target, kills, killer, death_log)
        cav_after = sum(c["cavalry"] for c in battle.contributions if c["faction"] == target)
        cav_died = cav_before - cav_after
        for _ in range(cav_died):
            death_log.append({"faction": target, "unit_type": "cavalry_death_ability_pending", "count": 1, "killer": killer})

    # cavalry death ability rolls
    ability_log = []
    for entry in death_log:
        if entry["unit_type"] == "cavalry_death_ability_pending":
            triggers = _roll_d20(rng) >= 14
            ability_log.append({"faction": entry["faction"], "free_infantry": triggers})

    # credit killers with kill-xp tokens for real unit deaths (not the synthetic ability marker)
    for entry in death_log:
        if entry["unit_type"] == "cavalry_death_ability_pending":
            continue
        # killer credit handled by caller via death_log directly

    battle.round_number += 1
    return death_log, ability_log


def is_battle_over(battle):
    return len(faction_alive_totals(battle)) <= 1


def get_winner(battle):
    alive = faction_alive_totals(battle)
    if len(alive) == 1:
        return next(iter(alive))
    return None  # mutual annihilation, or battle not over yet


def resolve_full_battle(battle, target_fn, players, rng=None):
    """Runs the whole battle to completion.

    target_fn(battle, faction) -> target_faction_or_None, called once
    per living faction per round (this is the agent decision point).
    players: {faction: PlayerState} - used to credit kill-XP tokens
    and queue the cavalry death-ability's free infantry.

    Returns a full log of every round's events, for replay/debugging.
    """
    rng = rng or random.Random()
    full_log = {"archer_phase": [], "rounds": []}

    archer_log = apply_archer_abilities(battle, rng)
    full_log["archer_phase"] = archer_log
    for entry in archer_log:
        if entry["killer"] in players:
            players[entry["killer"]].kill_xp_bank.append({"unit_type": entry["unit_type"]})

    rounds_run = 0
    while not is_battle_over(battle) and rounds_run < MAX_ROUNDS_SAFETY_CAP:
        target_choices = {}
        for faction in battle.factions():
            totals = battle.faction_totals()
            if sum(totals.get(faction, {}).values()) <= 0:
                continue
            target_choices[faction] = target_fn(battle, faction)

        death_log, ability_log = resolve_round(battle, target_choices, rng)
        full_log["rounds"].append({"deaths": death_log, "abilities": ability_log})

        for entry in death_log:
            if entry["unit_type"] == "cavalry_death_ability_pending":
                continue
            killer = entry["killer"]
            if killer in players:
                players[killer].kill_xp_bank.append({"unit_type": entry["unit_type"]})

        for entry in ability_log:
            if entry["free_infantry"] and entry["faction"] in players:
                players[entry["faction"]].pending_free_infantry += 1

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
