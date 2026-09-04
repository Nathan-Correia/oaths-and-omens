"""
Collect phase for engine - the last step of each turn (Buy -> Movement ->
Combat -> Collect), replacing the old start-of-turn income phase. Because
this now runs at the END of a turn, the following turn's Buy phase always
spends whatever gold/resources THIS Collect phase produces - turn 1's Buy
phase only has each faction's starting gold to spend, since Collect hasn't
run yet at that point (mirrors the rulebook's setup-phase carve-out: initial
gold/kill-XP are "awarded once at the very first buy phase instead of that
turn's regular income").

Three things happen here, in order, all folded into one apply_collect_phase
entrypoint (kept as separate functions since each is independently
testable): gold income, resource income, and the recurring per-round
victory-point tally (moved here from engine/turn.py - conceptually part of
the same phase, and doing so keeps VP_TO_WIN/OUTPOST_DESTROY_VP's home next
to the formula that PARTIALLY computes VP, so turn.py's battle-phase VP
award for destroying an outpost is the only other place a bit of VP logic
still lives).
"""

import numpy as np

from .state import NO_FACTION, RESOURCE_TO_INDEX, TERRAIN_TO_INDEX, UPGRADE_TO_INDEX

MOUNTAIN_INDEX = TERRAIN_TO_INDEX["mountain"]
LAKE_INDEX = TERRAIN_TO_INDEX["lake"]
PLAINS_INDEX = TERRAIN_TO_INDEX["plains"]
MARSH_INDEX = TERRAIN_TO_INDEX["marsh"]

BARRACKS_INDEX = UPGRADE_TO_INDEX["barracks"]
WORKSHOP_INDEX = UPGRADE_TO_INDEX["workshop"]
TEMPLE_INDEX = UPGRADE_TO_INDEX["temple"]

WOOD_INDEX = RESOURCE_TO_INDEX["wood"]
IRON_INDEX = RESOURCE_TO_INDEX["iron"]
CLAY_INDEX = RESOURCE_TO_INDEX["clay"]
FISH_INDEX = RESOURCE_TO_INDEX["fish"]

CAPITAL_GOLD = 3
OUTPOST_GOLD = 1
OUTPOST_GOLD_WITH_BARRACKS = 2

VP_TO_WIN = 50
OUTPOST_VP_PER_ROUND = 1
OUTPOST_DESTROY_VP = 2
TEMPLE_VP_PER_ROUND = 1


def apply_gold_income(state):
    """+3 gold/turn from a faction's capital, +1 per outpost (+2 instead of
    1 if that outpost has a Barracks upgrade - see UPGRADE_TYPES). Replaces
    (not stacks with) the old "3 + 1 per city beyond the 2nd" formula. A
    faction with zero cities gets no income and instead loses one unit (see
    _remove_first_unit) - unreachable in practice since capitals are
    permanent/uncapturable, kept only for parity with the pre-rulebook-
    update behavior."""
    for faction in range(state.num_factions):
        own = state.city_owner == faction
        if not np.any(own):
            state.gold[faction] = 0
            _remove_first_unit(state, faction)
            continue

        gold = CAPITAL_GOLD if np.any(own & state.is_capital) else 0
        outposts = np.nonzero(own & ~state.is_capital)[0]
        has_barracks = state.outpost_upgrade[outposts] == BARRACKS_INDEX
        gold += int(np.sum(np.where(has_barracks, OUTPOST_GOLD_WITH_BARRACKS, OUTPOST_GOLD)))
        state.gold[faction] += gold

    return state


def _remove_first_unit(state, faction):
    """Removes one unit (infantry -> cavalry -> archers priority) from the
    FIRST hex (in board/hex-index order) holding a peaceful army for
    `faction` with any units - only looks at peaceful board armies, never
    units currently locked in a pending battle."""
    candidates = np.nonzero((state.army_faction == faction) & (state.army_units.sum(axis=1) > 0))[0]
    if len(candidates) == 0:
        return
    hex_index = int(candidates[0])

    for ut in range(3):  # infantry, cavalry, archers priority
        if state.army_units[hex_index, ut] > 0:
            state.army_units[hex_index, ut] -= 1
            break

    if int(state.army_units[hex_index].sum()) == 0:
        state.army_faction[hex_index] = NO_FACTION
        state.frozen[hex_index] = False


def _outpost_resource(state, hex_index, faction, decide_resource_choice):
    """Which resource index (see RESOURCE_TO_INDEX) `hex_index` produces
    this turn, or None if it produces nothing (a desert outpost with no
    adjacent mountain or lake). Mountain/lake adjacency on any of the
    outpost's 6 neighbors overrides its own tile's resource; adjacency to
    both asks decide_resource_choice which one the player wants this turn.
    An outpost only ever produces ONE resource, regardless of how many
    qualifying neighbors it has."""
    neighbors = state.grid.neighbor_table[hex_index]
    valid = neighbors >= 0
    safe = np.where(valid, neighbors, 0)
    neighbor_terrain = state.terrain[safe]
    has_mountain = bool(np.any(valid & (neighbor_terrain == MOUNTAIN_INDEX)))
    has_lake = bool(np.any(valid & (neighbor_terrain == LAKE_INDEX)))

    if has_mountain and has_lake:
        choice = decide_resource_choice[faction](state, faction, hex_index)
        return IRON_INDEX if choice == "iron" else FISH_INDEX
    if has_mountain:
        return IRON_INDEX
    if has_lake:
        return FISH_INDEX

    own_terrain = int(state.terrain[hex_index])
    if own_terrain == PLAINS_INDEX:
        return WOOD_INDEX
    if own_terrain == MARSH_INDEX:
        return CLAY_INDEX
    return None


def apply_resource_income(state, decide_resource_choice):
    """Generates Wood/Iron/Clay/Fish for every outpost - capitals never
    generate resources. decide_resource_choice: {faction: (state, faction,
    hex_index) -> "iron" | "fish"}, only ever consulted for an outpost
    adjacent to both a mountain and a lake (see _outpost_resource)."""
    for faction in range(state.num_factions):
        outposts = np.nonzero((state.city_owner == faction) & ~state.is_capital)[0]
        for hex_index in outposts:
            hex_index = int(hex_index)
            resource = _outpost_resource(state, hex_index, faction, decide_resource_choice)
            if resource is None:
                continue
            amount = 2 if state.outpost_upgrade[hex_index] == WORKSHOP_INDEX else 1
            state.resources[faction, resource] += amount

    return state


def apply_victory_points(state):
    """End-of-round VP tally (the win condition - see turn.py's
    get_game_winner): your first outpost earns nothing, and each additional
    one beyond that earns OUTPOST_VP_PER_ROUND more - so 1 outpost is worth
    0/round, 2 is worth 1, 3 is worth 2, and so on (max(0, outposts - 1),
    not a flat per-outpost rate) - plus a flat TEMPLE_VP_PER_ROUND for every
    outpost that has a Temple upgrade, on top of that formula. Capitals
    don't count. Destroying an enemy outpost is awarded separately,
    immediately, in turn.py's _run_battle_phase (OUTPOST_DESTROY_VP) - this
    only covers the recurring per-round income."""
    for faction in range(state.num_factions):
        own_outposts = (state.city_owner == faction) & ~state.is_capital
        outposts = int(np.sum(own_outposts))
        temples = int(np.sum(state.outpost_upgrade[own_outposts] == TEMPLE_INDEX))
        state.victory_points[faction] += (
            max(0, outposts - 1) * OUTPOST_VP_PER_ROUND + temples * TEMPLE_VP_PER_ROUND
        )

    return state


def apply_collect_phase(state, decide_resource_choice):
    """Runs the full Collect phase in place: gold income, resource income,
    then the per-round VP tally (each is independently testable above -
    see this module's docstring for why they're bundled into one call from
    turn.py)."""
    apply_gold_income(state)
    apply_resource_income(state, decide_resource_choice)
    apply_victory_points(state)
    return state
