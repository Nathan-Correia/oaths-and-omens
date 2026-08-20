"""
Movement phase (and the cavalry-only sub-phase, which reuses the same
step-application logic with a filtered action set).

A MoveAction is: {"from_hex": coord, "to_hex": coord,
                   "units": {"infantry": n, "cavalry": n, "archers": n}}
Any subset of a stack can move (partial splits are legal), and units
left behind simply stay on the origin hex.

ACTION-SPACE SHAPING: get_legal_movement_actions/get_legal_cavalry_actions
used to enumerate every non-empty sub-multiset split of every eligible
army, times every direction, every step (see _all_splits below) -
profiling showed this combinatorial blow-up (unit-split x direction x
army) as the single largest cost in the engine. In practice neither
SmartRandomAgent nor HeuristicAgent ever choose anything but the full
stack (they rank candidate actions by total units moved, and the
full-stack split is always the unique maximum), so the split
enumeration was pure waste for every non-fuzzing agent - and it's also
an unusably large discrete action space for an eventual RL policy.

Both functions now return exactly one action per (eligible army,
passable neighbor): move the whole stack (or, in the cavalry phase,
the whole cavalry contingent). This is a strict subset of the old
output, so any code that always picked the max-units action for a
given destination (SmartRandomAgent, HeuristicAgent) sees identical
behavior. RandomAgent - whose job is to fuzz partial-split code paths,
not play well - now does its own on-demand sub-splitting (see
agents/random_agent.py) on the single action it picks, rather than the
engine materializing every possible split up front for every army.
_all_splits() is kept around for that purpose (and for tests/tools that
want the full split enumeration for a single army).

RULE: only one army may move per player per step - matches the
physical "place one hand on one army and move it" mechanic. Enforced
in _validate_and_collect: if an agent submits more than one action in
a single step, only the first is considered; the rest are ignored.

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

Overstacking outside of battles: a peaceful merge that would push a
stack past the 6-unit cap simply doesn't happen - the move is treated
as illegal and the army stays at its origin, untouched. This can only
be decided at apply-time (not in get_legal_movement_actions), because
whether a given hex ends up hostile/peaceful/overflowing depends on
every player's simultaneous choices that step, not just one player's.
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
        full_units = {"infantry": army["infantry"], "cavalry": army["cavalry"], "archers": army["archers"]}
        for n in hex_neighbors(coord, state.radius):
            nh = state.board.get(n)
            if not nh or nh.terrain in IMPASSABLE_TERRAIN:
                continue
            actions.append({"from_hex": coord, "to_hex": n, "units": dict(full_units)})
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
            actions.append({"from_hex": coord, "to_hex": n,
                             "units": {"infantry": 0, "cavalry": army["cavalry"], "archers": 0}})
    return actions


def _validate_and_collect(state, actions_by_faction):
    """Returns a list of validated moves: {"faction", "from_hex", "to_hex", "units"}.

    Enforces "one army per player per step": only the FIRST action in
    each faction's submitted list is considered at all - if an agent
    submits more than one, the rest are simply ignored, mirroring the
    physical "place one hand on one army" rule. Invalid moves (bad
    origin, insufficient units, locked/frozen origin, impassable
    destination) are silently dropped."""
    collected = []
    for faction, actions in actions_by_faction.items():
        if not actions:
            continue
        action = actions[0]
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
            # two different factions attacking directly into each other -> line battle.
            # Departures must be applied BEFORE _start_or_extend_battle: the
            # battle hex is chosen as min(from_hex, to_hex), which is always
            # one of the two swap origins - and _start_or_extend_battle
            # immediately clears that hex's army. Subtracting departures
            # afterward would find that origin's army already wiped out
            # from under it whenever battle_hex happened to equal it.
            for mm in (m, reverse):
                origin = state.board[mm["from_hex"]]
                _subtract_units(origin.army, mm["units"])
                if army_total(origin.army) == 0:
                    origin.army = None

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

    # --- pass 2: group remaining (non-swap) moves by destination.
    # Departures are applied upfront for every remaining move (like the
    # swap pass above) - this has to happen before any per-destination
    # classification below, because a hex can simultaneously be one
    # faction's own outbound origin AND another faction's incoming-
    # attack destination (which locks/clears it into a battle); if
    # departure were deferred, whichever gets processed second could
    # find its origin's army already cleared out from under it.
    # The one case this complicates is the overstack rejection just
    # below: since departure already happened, an illegal overstacked
    # merge is undone by reverting (adding the units back) rather than
    # simply not applying it in the first place.
    arrivals_by_dest = {}
    for m in remaining_moves:
        arrivals_by_dest.setdefault(m["to_hex"], []).append(m)
        origin = state.board[m["from_hex"]]
        _subtract_units(origin.army, m["units"])
        if army_total(origin.army) == 0:
            origin.army = None

    def _revert_departure(a):
        """Used only when a peaceful merge is rejected for overstacking:
        sends the units back to where they came from. If that origin
        hex has since been claimed by a different faction (a rare
        knock-on collision from another simultaneous move this same
        step), the returning units fight rather than vanish or silently
        merge into someone else's army."""
        origin = state.board[a["from_hex"]]
        if origin.army is None:
            origin.army = {"faction": a["faction"], "infantry": 0, "cavalry": 0, "archers": 0, "frozen": False}
            _add_units(origin.army, a["units"])
        elif origin.army["faction"] == a["faction"]:
            _add_units(origin.army, a["units"])
        else:
            contributions = [
                {"faction": origin.army["faction"], "origin_hex": a["from_hex"],
                 **{ut: origin.army[ut] for ut in UNIT_TYPES}},
                {"faction": a["faction"], "origin_hex": a["from_hex"], **a["units"]},
            ]
            _start_or_extend_battle(state, a["from_hex"], contributions)

    for dest, arrivals in arrivals_by_dest.items():
        dh = state.board[dest]

        if dh.locked:
            # reinforcing an already-pending battle - always succeeds,
            # battles are allowed to exceed the stack cap mid-fight
            contributions = [{"faction": a["faction"], "origin_hex": a["from_hex"], **a["units"]} for a in arrivals]
            _start_or_extend_battle(state, dest, contributions)
            continue

        arrival_factions = {a["faction"] for a in arrivals}
        existing_faction = dh.army["faction"] if dh.army else None

        hostile_present = existing_faction is not None and existing_faction not in arrival_factions
        multiple_arrival_factions = len(arrival_factions) > 1

        if hostile_present or multiple_arrival_factions:
            # battle - always succeeds regardless of cap
            contributions = [{"faction": a["faction"], "origin_hex": a["from_hex"], **a["units"]} for a in arrivals]
            if dh.army is not None:
                contributions.append({"faction": existing_faction, "origin_hex": dest, **{
                    ut: dh.army[ut] for ut in UNIT_TYPES
                }})
            _start_or_extend_battle(state, dest, contributions)
        else:
            # peaceful merge - must respect the 6-unit stack cap. If it
            # doesn't fit, the move is illegal: revert it (units go back
            # to their origin - see _revert_departure). (With the one-
            # army-per-step rule, `arrivals` here is always from a single
            # faction, since a hex can only end up with multiple
            # simultaneous peaceful arrivals if they're hostile/multi-
            # faction, which is handled above instead.)
            existing_total = army_total(dh.army) if dh.army else 0
            arriving_total = sum(sum(a["units"].values()) for a in arrivals)
            if existing_total + arriving_total > MAX_STACK_SIZE:
                for a in arrivals:
                    _revert_departure(a)
                continue

            if dh.army is None:
                dh.army = empty_army(existing_faction if existing_faction is not None else next(iter(arrival_factions)))
            for a in arrivals:
                _add_units(dh.army, a["units"])

            if dh.terrain == "marsh":
                dh.army["frozen"] = True

            _maybe_capture_city(state, dest)

    return set(state.battles.keys())