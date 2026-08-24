"""
Movement mechanics for engine - ported from engine/movement.py to
operate on ArrayState instead of a dict-of-HexState board.

SCOPE decision: unlike engine/movement.py (which supports moving any
partial split of a stack, to preserve RandomAgent's fuzz-testing power),
engine actions only ever move a FIXED, phase-determined subset of a
hex's army: the whole thing during the movement phase, or just the
cavalry count (leaving 100% of any infantry/archers behind) during the
cavalry phase - never an arbitrary combinatorial split. This matches the
fixed (hex, direction) action-space design from the RL-design
conversation this was scoped from, and isn't actually a behavior change
relative to how the non-fuzzing agents (SmartRandomAgent/HeuristicAgent)
already play: engine/movement.py's get_legal_movement_actions/
get_legal_cavalry_actions already return exactly these two shapes (see
that module's docstring) - engine just bakes that in as the only
representation, rather than a convention every agent happens to follow.
Partial-split control (beyond the movement/cavalry distinction) can be
added back as an extra action dimension later if it turns out to matter.

Otherwise this mirrors engine/movement.py's apply_movement_step logic
step for step (swap/line-battle detection, then per-destination
grouping: reinforcing an already-locked hex, starting a battle on
hostile contact or multiple simultaneous arrivals, or a capped peaceful
merge with reversion on overstacking) - see that module's docstring for
the full rules rationale. Ported quirks and all (see _revert_departure's
docstring) since the goal is matching engine/'s actual behavior for
parity testing, not improving on it.

RULE CHANGE: capitals and outposts are no longer capturable by walking
into them undefended - arriving at a hex owned by a foreign faction
(city_owner set to someone other than every arriving faction) always
forces a battle now, the same as arriving on real hostile units, even
if the tile has no defending army at all. What happens to the winner
afterward (evicted with nothing to show for it at a capital, or left
standing on a now-ownerless destroyed outpost) is turn.py's
_run_battle_phase's job, not this module's - see its docstring.
"""

import numpy as np

from .state import IMPASSABLE_TERRAIN_INDICES, MAX_STACK_SIZE, NO_FACTION, TERRAIN_TO_INDEX

MARSH_INDEX = TERRAIN_TO_INDEX["marsh"]


def _legal_mask(state, faction, require_cavalry):
    grid = state.grid
    own_army = (state.army_faction == faction) & ~state.locked & ~state.frozen
    if require_cavalry:
        own_army = own_army & (state.army_units[:, 1] > 0)

    neighbor = grid.neighbor_table  # [N, 6]
    has_neighbor = neighbor >= 0
    safe_neighbor = np.where(has_neighbor, neighbor, 0)
    neighbor_terrain = state.terrain[safe_neighbor]
    passable = ~np.isin(neighbor_terrain, IMPASSABLE_TERRAIN_INDICES)

    return own_army[:, None] & has_neighbor & passable


def legal_movement_mask(state, faction):
    """[num_hexes, 6] bool: legal_mask[h, d] = True iff faction's whole
    army at hex h may move to h's neighbor in direction d this step."""
    return _legal_mask(state, faction, require_cavalry=False)


def legal_cavalry_mask(state, faction):
    """Same, but only for hexes with cavalry present - see
    engine/movement.py's get_legal_cavalry_actions."""
    return _legal_mask(state, faction, require_cavalry=True)


def _units_at(state, hex_index):
    return state.army_units[hex_index].copy()


def _units_total(units):
    return int(units.sum())


def _clear_army(state, hex_index):
    """Full wipe - used where the WHOLE army at a hex is absorbed into a
    battle (see _start_or_extend_battle). Not used for departures - see
    _subtract_departure, which only removes what actually moved."""
    state.army_faction[hex_index] = NO_FACTION
    state.army_units[hex_index] = 0
    state.frozen[hex_index] = False


def _subtract_departure(state, hex_index, units):
    """Removes `units` from hex_index's army - the whole thing for a
    movement-phase action, or just the cavalry count for a cavalry-phase
    one (see apply_movement_step's cavalry_only). Mirrors
    engine/movement.py's _subtract_units + "clear if now empty" check:
    a cavalry-only departure can leave infantry/archers behind, still
    owned by the same faction."""
    state.army_units[hex_index] = state.army_units[hex_index] - units
    if int(state.army_units[hex_index].sum()) == 0:
        state.army_faction[hex_index] = NO_FACTION
        state.frozen[hex_index] = False


def _first_empty_battle_slot(state, hex_index):
    used = state.battle_faction[hex_index] != NO_FACTION
    empty = np.nonzero(~used)[0]
    if len(empty) == 0:
        raise RuntimeError(
            f"hex {hex_index} battle contributions exceeded MAX_BATTLE_CONTRIB "
            f"({state.battle_faction.shape[1]}) - raise the cap"
        )
    return int(empty[0])


def _start_or_extend_battle(state, hex_index, contributions):
    """contributions: list of (faction, origin_index, units[3]). Mirrors
    engine/movement.py's _start_or_extend_battle: appends to whatever's
    already there (round number untouched if the hex was already
    locked, reset to 0 the moment it newly becomes locked), and clears/
    locks the hex's peaceful-army slot. Also records the hex into
    state.battle_order the moment it newly locks - see state.py's
    module docstring for why battle processing order matters."""
    was_locked = bool(state.locked[hex_index])
    for faction, origin_index, units in contributions:
        slot = _first_empty_battle_slot(state, hex_index)
        state.battle_faction[hex_index, slot] = faction
        state.battle_origin[hex_index, slot] = origin_index
        state.battle_units[hex_index, slot] = units
    if not was_locked:
        state.battle_round[hex_index] = 0
        state.battle_order.append(hex_index)
    state.locked[hex_index] = True
    _clear_army(state, hex_index)


def _units_to_move(state, from_index, cavalry_only):
    """The movement-phase action moves the WHOLE army at a hex (see
    module docstring). The cavalry-phase action is a different, fixed
    subset, not a further narrowing of that same idea: it moves only
    the cavalry count, always leaving 100% of any infantry/archers
    behind - matching engine/movement.py's get_legal_cavalry_actions,
    which builds its units dict as {"infantry": 0, "cavalry": N,
    "archers": 0} rather than the full stack."""
    if cavalry_only:
        units = np.zeros(3, dtype=state.army_units.dtype)
        units[1] = state.army_units[from_index, 1]
        return units
    return _units_at(state, from_index)


def _validate_and_collect(state, actions_by_faction, cavalry_only):
    """actions_by_faction: {faction: (from_index, direction) or None}.
    Returns a list of {"faction", "from", "to", "units"} for moves that
    are actually legal right now. Silently drops anything invalid,
    mirroring engine/movement.py's _validate_and_collect - "one action
    per faction" is already guaranteed by the dict shape here, unlike
    v1's list-based input which has to explicitly take only actions[0]."""
    grid = state.grid
    collected = []
    for faction, action in actions_by_faction.items():
        if action is None:
            continue
        from_index, direction = action

        if state.army_faction[from_index] != faction:
            continue
        if state.locked[from_index] or state.frozen[from_index]:
            continue
        units = _units_to_move(state, from_index, cavalry_only)
        if units.sum() <= 0:
            continue

        to_index = grid.neighbor_table[from_index, direction]
        if to_index < 0:
            continue
        if state.terrain[to_index] in IMPASSABLE_TERRAIN_INDICES:
            continue

        collected.append({"faction": faction, "from": int(from_index), "to": int(to_index), "units": units})
    return collected


def apply_movement_step(state, actions_by_faction, cavalry_only=False):
    """Applies one simultaneous movement step in place. Returns the set
    of hex indices with a battle pending after this step (new or
    reinforced) - same contract as engine/movement.py's
    apply_movement_step, which this same function serves for both the
    regular and cavalry movement phases (the caller decides which by
    which legal-action generator it used - see turn.py in engine/).
    `cavalry_only` must match: True for actions built from
    legal_cavalry_mask, False for legal_movement_mask."""
    moves = _validate_and_collect(state, actions_by_faction, cavalry_only)

    # --- pass 1: swap / line-battle detection ---
    swap_pairs_handled = set()
    remaining_moves = []
    moves_by_from_to = {(m["from"], m["to"]): m for m in moves}

    for m in moves:
        key = (m["from"], m["to"])
        reverse_key = (m["to"], m["from"])
        if key in swap_pairs_handled or reverse_key in swap_pairs_handled:
            continue
        reverse = moves_by_from_to.get(reverse_key)
        if reverse is not None and reverse["faction"] != m["faction"]:
            for mm in (m, reverse):
                _subtract_departure(state, mm["from"], mm["units"])

            battle_hex = min(m["from"], m["to"])
            contributions = [
                (m["faction"], m["from"], m["units"]),
                (reverse["faction"], reverse["from"], reverse["units"]),
            ]
            _start_or_extend_battle(state, battle_hex, contributions)
            swap_pairs_handled.add(key)
            swap_pairs_handled.add(reverse_key)
        else:
            remaining_moves.append(m)

    # --- pass 2: group remaining moves by destination ---
    arrivals_by_dest = {}
    for m in remaining_moves:
        arrivals_by_dest.setdefault(m["to"], []).append(m)
        _subtract_departure(state, m["from"], m["units"])

    def _revert_departure(a):
        """Sends a's units back to its origin. Ported as-is from
        engine/movement.py's _revert_departure, including its edge-case
        behavior if the origin has since been claimed by a different
        faction's peaceful merge (starts a battle there instead of
        vanishing/merging silently) or - a latent quirk present in v1
        too, not introduced here - become locked by an unrelated battle
        this same step (recreates a peaceful army on a locked hex)."""
        origin = a["from"]
        if state.army_faction[origin] == NO_FACTION:
            state.army_faction[origin] = a["faction"]
            state.army_units[origin] = a["units"]
        elif state.army_faction[origin] == a["faction"]:
            state.army_units[origin] = state.army_units[origin] + a["units"]
        else:
            contributions = [
                (int(state.army_faction[origin]), origin, _units_at(state, origin)),
                (a["faction"], origin, a["units"]),
            ]
            _start_or_extend_battle(state, origin, contributions)

    for dest, arrivals in arrivals_by_dest.items():
        if state.locked[dest]:
            contributions = [(a["faction"], a["from"], a["units"]) for a in arrivals]
            _start_or_extend_battle(state, dest, contributions)
            continue

        arrival_factions = {a["faction"] for a in arrivals}
        existing_faction = int(state.army_faction[dest]) if state.army_faction[dest] != NO_FACTION else None

        hostile_present = existing_faction is not None and existing_faction not in arrival_factions
        multiple_arrival_factions = len(arrival_factions) > 1
        dest_owner = int(state.city_owner[dest]) if state.city_owner[dest] != NO_FACTION else None
        foreign_structure = dest_owner is not None and dest_owner not in arrival_factions

        if hostile_present or multiple_arrival_factions or foreign_structure:
            contributions = [(a["faction"], a["from"], a["units"]) for a in arrivals]
            if existing_faction is not None:
                contributions.append((existing_faction, dest, _units_at(state, dest)))
            _start_or_extend_battle(state, dest, contributions)
        else:
            existing_total = _units_total(state.army_units[dest]) if existing_faction is not None else 0
            arriving_total = sum(_units_total(a["units"]) for a in arrivals)
            if existing_total + arriving_total > MAX_STACK_SIZE:
                for a in arrivals:
                    _revert_departure(a)
                continue

            resolved_faction = existing_faction if existing_faction is not None else next(iter(arrival_factions))
            state.army_faction[dest] = resolved_faction
            for a in arrivals:
                state.army_units[dest] = state.army_units[dest] + a["units"]

            if state.terrain[dest] == MARSH_INDEX:
                state.frozen[dest] = True

    return {int(i) for i in np.nonzero(state.locked)[0]}
