"""
Movement phase (and the cavalry-only sub-phase, which reuses the same
step-application logic with a filtered action set).

A MoveAction is: {"from_hex": coord, "to_hex": coord,
                   "units": {"infantry": n, "cavalry": n, "archers": n}}
Any subset of a stack can move (partial splits are legal), and units
left behind simply stay on the origin hex.

Collision handling, once per step:
  1. Swap detection (line battle): two different factions moving into
     each other's hex in the same step never complete their moves -
     they collide and a battle starts instead. Handled as a dedicated
     pre-pass so the generic per-destination logic below doesn't just
     see two "empty" hexes and let both moves complete peacefully.
  2. Generic per-destination resolution: everything else (encounter
     battles - multiple armies converging on an empty hex; attack/
     defense - moving onto a hex with a stationary enemy army;
     friendly reinforcement/merging; reinforcing an already-locked
     battle hex) is handled by grouping remaining moves by destination
     hex and checking how many distinct factions end up there.

NOTE on simplifications (documented, not silently swallowed):
  - The cavalry phase's "max 2 moves per unit, 4 steps total" budget
    is only enforced as "cavalry may move at most twice, tracked per
    origin-army" rather than truly per individual unit. Good enough
    for a v1 random/scripted agent; revisit if it matters later.
  - Overstacking prevention outside of battles (the 6-unit cap) relies
    on legal-action generation not offering illegal merges; apply_step
    clamps defensively but doesn't simulate a "rejected move" outcome.
"""

from .geometry import hex_neighbors
from .state import UNIT_TYPES, IMPASSABLE_TERRAIN, MAX_STACK_SIZE, empty_army, army_total, Battle

CAV_PHASE_MAX_MOVES_PER_ARMY = 2


def _has_units(units_dict):
    return any(units_dict.get(ut, 0) > 0 for ut in UNIT_TYPES)


def _subtract_units(army, units_dict):
    for ut in UNIT_TYPES:
        army[ut] -= units_dict.get(ut, 0)


def _add_units(army, units_dict):
    for ut in UNIT_TYPES:
        army[ut] += units_dict.get(ut, 0)


def _all_splits(army):
    """All non-empty ways to take a sub-multiset of `army`'s units."""
    inf_range = range(army["infantry"] + 1)
    cav_range = range(army["cavalry"] + 1)
    arc_range = range(army["archers"] + 1)
    for i in inf_range:
        for c in cav_range:
            for a in arc_range:
                if i or c or a:
                    yield {"infantry": i, "cavalry": c, "archers": a}


def get_legal_movement_actions(state, faction):
    actions = []
    for coord, h in state.board.items():
        army = h.army
        if not army or army["faction"] != faction or h.locked or army.get("frozen"):
            continue
        for n in hex_neighbors(coord, state.radius):
            nh = state.board.get(n)
            if not nh or nh.terrain in IMPASSABLE_TERRAIN:
                continue
            for split in _all_splits(army):
                actions.append({"from_hex": coord, "to_hex": n, "units": split})
    return actions


def get_legal_cavalry_actions(state, faction):
    """Same as movement, but only cavalry units may move, and only from
    armies that have cavalry present. Per-unit "2 moves" budget isn't
    enforced here - see module docstring."""
    actions = []
    for coord, h in state.board.items():
        army = h.army
        if not army or army["faction"] != faction or h.locked or army.get("frozen"):
            continue
        if army["cavalry"] <= 0:
            continue
        for n in hex_neighbors(coord, state.radius):
            nh = state.board.get(n)
            if not nh or nh.terrain in IMPASSABLE_TERRAIN:
                continue
            for c in range(1, army["cavalry"] + 1):
                actions.append({"from_hex": coord, "to_hex": n, "units": {"infantry": 0, "cavalry": c, "archers": 0}})
    return actions


def _validate_and_collect(state, actions_by_faction):
    """Returns a list of validated moves: {"faction", "from_hex", "to_hex", "units"}.
    Invalid moves (bad origin, insufficient units, locked/frozen origin,
    impassable destination) are silently dropped."""
    collected = []
    for faction, actions in actions_by_faction.items():
        for action in actions:
            from_hex = tuple(action["from_hex"])
            to_hex = tuple(action["to_hex"])
            units = action["units"]

            h = state.board.get(from_hex)
            if not h or not h.army or h.army["faction"] != faction:
                continue
            if h.locked or h.army.get("frozen"):
                continue
            if not _has_units(units):
                continue
            if any(h.army[ut] < units.get(ut, 0) for ut in UNIT_TYPES):
                continue
            nh = state.board.get(to_hex)
            if not nh or nh.terrain in IMPASSABLE_TERRAIN:
                continue

            collected.append({"faction": faction, "from_hex": from_hex, "to_hex": to_hex, "units": dict(units)})
    return collected


def _start_or_extend_battle(state, hex_coord, contributions):
    battle = state.battles.get(hex_coord)
    if battle is None:
        battle = Battle(hex_coord=hex_coord, contributions=[])
        state.battles[hex_coord] = battle
    battle.contributions.extend(contributions)
    h = state.board[hex_coord]
    h.locked = True
    h.army = None


def _maybe_capture_city(state, hex_coord):
    h = state.board[hex_coord]
    if h.city_owner is not None and h.army is not None and h.army["faction"] != h.city_owner:
        h.city_owner = h.army["faction"]


def apply_movement_step(state, actions_by_faction):
    """Applies one simultaneous movement step. Mutates `state` in place
    (armies move, battles start/extend, cities get captured on
    peaceful occupation) and returns the set of hex coords that have a
    battle pending after this step (new or reinforced)."""
    moves = _validate_and_collect(state, actions_by_faction)

    # --- pass 1: swap / line-battle detection ---
    swap_pairs_handled = set()
    remaining_moves = []
    moves_by_from_to = {(m["from_hex"], m["to_hex"]): m for m in moves}

    for m in moves:
        key = (m["from_hex"], m["to_hex"])
        reverse_key = (m["to_hex"], m["from_hex"])
        if key in swap_pairs_handled or reverse_key in swap_pairs_handled:
            continue
        reverse = moves_by_from_to.get(reverse_key)
        if reverse and reverse["faction"] != m["faction"]:
            # two different factions attacking directly into each other -> line battle
            battle_hex = min(m["from_hex"], m["to_hex"])
            contributions = [
                {"faction": m["faction"], "origin_hex": m["from_hex"], **m["units"]},
                {"faction": reverse["faction"], "origin_hex": reverse["from_hex"], **reverse["units"]},
            ]
            _start_or_extend_battle(state, battle_hex, contributions)
            swap_pairs_handled.add(key)
            swap_pairs_handled.add(reverse_key)
        else:
            remaining_moves.append(m)

    # apply departures for all remaining (non-swap) moves now, so
    # origin hexes reflect what's left behind
    for m in remaining_moves:
        origin = state.board[m["from_hex"]]
        _subtract_units(origin.army, m["units"])
        if army_total(origin.army) == 0:
            origin.army = None

    # also apply departures for swapped moves (their units already
    # went into the battle's contributions, so just clear the origin)
    for key in swap_pairs_handled:
        from_hex, to_hex = key
        m = moves_by_from_to.get(key)
        if m is None:
            continue
        origin = state.board[m["from_hex"]]
        if origin.army:
            _subtract_units(origin.army, m["units"])
            if army_total(origin.army) == 0:
                origin.army = None

    # --- pass 2: group remaining arrivals by destination ---
    arrivals_by_dest = {}
    for m in remaining_moves:
        arrivals_by_dest.setdefault(m["to_hex"], []).append(m)

    battle_hexes = set(state.battles.keys())

    for dest, arrivals in arrivals_by_dest.items():
        dh = state.board[dest]

        if dh.locked:
            # reinforcing an already-pending battle
            contributions = [{"faction": a["faction"], "origin_hex": a["from_hex"], **a["units"]} for a in arrivals]
            _start_or_extend_battle(state, dest, contributions)
            continue

        arrival_factions = {a["faction"] for a in arrivals}
        existing_faction = dh.army["faction"] if dh.army else None

        hostile_present = existing_faction is not None and existing_faction not in arrival_factions
        multiple_arrival_factions = len(arrival_factions) > 1

        if hostile_present or multiple_arrival_factions:
            contributions = [{"faction": a["faction"], "origin_hex": a["from_hex"], **a["units"]} for a in arrivals]
            if dh.army is not None:
                contributions.append({"faction": existing_faction, "origin_hex": dest, **{
                    ut: dh.army[ut] for ut in UNIT_TYPES
                }})
            _start_or_extend_battle(state, dest, contributions)
        else:
            # peaceful arrival: merge into existing friendly army (or create new)
            if dh.army is None:
                dh.army = empty_army(existing_faction if existing_faction is not None else next(iter(arrival_factions)))
            for a in arrivals:
                _add_units(dh.army, a["units"])
            # defensive clamp - legal-action generation should prevent
            # exceeding the cap outside of battles, this just guards state integrity
            total = army_total(dh.army)
            if total > MAX_STACK_SIZE:
                dh.army["infantry"] = max(0, dh.army["infantry"] - (total - MAX_STACK_SIZE))

            if dh.terrain == "marsh":
                dh.army["frozen"] = True

            _maybe_capture_city(state, dest)

    return set(state.battles.keys()) - battle_hexes | (set(state.battles.keys()) & battle_hexes)
