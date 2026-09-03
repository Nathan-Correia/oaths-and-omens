"""
Batched Collect phase for engine - the last step of each turn (Buy ->
Movement -> Combat -> Collect), across every game in the batch at once.
Because this runs at the END of a turn, the following turn's Buy phase
always spends whatever gold/resources THIS Collect phase produces - turn
1's Buy phase only has each faction's starting gold to spend, since
Collect hasn't run yet at that point (mirrors the rulebook's setup-phase
carve-out).

Three things happen here, in order, all folded into one
apply_collect_phase entrypoint (kept as separate functions since each is
independently testable): gold income, resource income, and the recurring
per-round victory-point tally.

Fully vectorized except one deliberately-unbatched sliver: an outpost
adjacent to BOTH a mountain and a lake asks its owning faction's agent
which resource to take (decide_resource_choice), the one genuine decision
point in this whole phase - agents stay per-game Python functions (see
the plan's "Consequence for agents/tooling" section), so resolving these
(typically rare - most outposts aren't adjacent to both) is a small
Python loop over however many such cases actually occur this collect
phase, not a loop over every hex or every game.
"""

import torch

from .state import NO_FACTION, RESOURCE_TO_INDEX, TERRAIN_TO_INDEX, UPGRADE_TO_INDEX, unstack_states

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


def _ownership_mask(state):
    """[B, N, F] bool: own[b, h, f] = True iff faction f owns hex h in
    game b (capital or outpost, same as city_owner==f)."""
    F = state.num_factions
    faction_ids = torch.arange(F, device=state.device)
    return state.city_owner.unsqueeze(-1) == faction_ids


def apply_gold_income(state):
    """+3 gold/turn from a faction's capital, +1 per outpost (+2 instead
    of 1 with a Barracks upgrade). A faction with zero cities gets no
    income and instead loses one unit (see _remove_first_unit_batched) -
    unreachable in practice since capitals are permanent/uncapturable,
    kept only for parity with the pre-rulebook-update behavior (and, same
    as that behavior, handled with a small non-vectorized loop since it
    essentially never fires)."""
    own = _ownership_mask(state)  # [B, N, F]
    has_any_city = own.any(dim=1)  # [B, F]
    has_capital = (own & state.is_capital.unsqueeze(-1)).any(dim=1)  # [B, F]
    gold_from_capital = torch.where(has_capital, CAPITAL_GOLD, 0)

    is_outpost_owned = own & (~state.is_capital).unsqueeze(-1)  # [B, N, F]
    has_barracks = (state.outpost_upgrade == BARRACKS_INDEX).unsqueeze(-1)  # [B, N, 1]
    per_hex_gold = torch.where(has_barracks, OUTPOST_GOLD_WITH_BARRACKS, OUTPOST_GOLD)
    outpost_gold = (is_outpost_owned * per_hex_gold).sum(dim=1)  # [B, F]

    total = (gold_from_capital + outpost_gold).to(state.gold.dtype)
    state.gold += torch.where(has_any_city, total, torch.zeros_like(total))

    no_city = ~has_any_city
    if bool(no_city.any()):
        state.gold[no_city] = 0
        _remove_first_unit_batched(state, no_city)
    return state


def _remove_first_unit_batched(state, target_mask):
    """target_mask: [B, F] bool - for each flagged (batch, faction),
    removes one unit (infantry -> cavalry -> archers priority) from the
    FIRST hex (lowest hex index) holding a peaceful army for that faction
    with any units - only peaceful board armies, never units locked in a
    pending battle. A small Python loop over target_mask's (rare - see
    apply_gold_income) True entries, not over every hex/game."""
    b_idx, f_idx = torch.nonzero(target_mask, as_tuple=True)
    for i in range(len(b_idx)):
        b, f = int(b_idx[i]), int(f_idx[i])
        candidates = torch.nonzero(
            (state.army_faction[b] == f) & (state.army_units[b].sum(dim=-1) > 0), as_tuple=False
        ).flatten()
        if len(candidates) == 0:
            continue
        h = int(candidates[0])
        for ut in range(3):
            if state.army_units[b, h, ut] > 0:
                state.army_units[b, h, ut] -= 1
                break
        if int(state.army_units[b, h].sum()) == 0:
            state.army_faction[b, h] = NO_FACTION
            state.frozen[b, h] = False


def apply_resource_income(state, decide_resource_choice_list):
    """Generates Wood/Iron/Clay/Fish for every outpost - capitals never
    generate resources. decide_resource_choice_list: length B, one
    {faction: (state_b, faction, hex_index) -> "iron" | "fish"} dict per
    batch item, matching how the batched engine represents per-game agent
    policies elsewhere (see movement.py's actions_from_dicts) - only ever
    consulted for an outpost adjacent to both a mountain and a lake (see
    module docstring)."""
    B, N = state.terrain.shape
    device = state.device
    grid = state.grid

    neighbor = grid.neighbor_table  # [N, 6]
    has_neighbor = neighbor >= 0
    safe_neighbor = torch.where(has_neighbor, neighbor, torch.zeros_like(neighbor)).long()
    neighbor_terrain = state.terrain[:, safe_neighbor]  # [B, N, 6]
    has_mountain = (has_neighbor[None, :, :] & (neighbor_terrain == MOUNTAIN_INDEX)).any(dim=-1)  # [B, N]
    has_lake = (has_neighbor[None, :, :] & (neighbor_terrain == LAKE_INDEX)).any(dim=-1)  # [B, N]
    ambiguous = has_mountain & has_lake

    resource = torch.full((B, N), -1, dtype=torch.long, device=device)
    resource = torch.where(has_mountain, torch.full_like(resource, IRON_INDEX), resource)
    resource = torch.where(~has_mountain & has_lake, torch.full_like(resource, FISH_INDEX), resource)
    own_terrain_only = ~has_mountain & ~has_lake
    resource = torch.where(own_terrain_only & (state.terrain == PLAINS_INDEX), torch.full_like(resource, WOOD_INDEX), resource)
    resource = torch.where(own_terrain_only & (state.terrain == MARSH_INDEX), torch.full_like(resource, CLAY_INDEX), resource)

    valid_outpost = (state.city_owner != NO_FACTION) & ~state.is_capital  # [B, N]

    amb_b, amb_h = torch.nonzero(ambiguous & valid_outpost, as_tuple=True)
    if len(amb_b):
        state_views = unstack_states(state)
        for i in range(len(amb_b)):
            b, h = int(amb_b[i]), int(amb_h[i])
            f = int(state.city_owner[b, h])
            choice = decide_resource_choice_list[b][f](state_views[b], f, h)
            resource[b, h] = IRON_INDEX if choice == "iron" else FISH_INDEX

    valid = valid_outpost & (resource >= 0)
    b_idx, h_idx = torch.nonzero(valid, as_tuple=True)
    if len(b_idx):
        f_idx = state.city_owner[b_idx, h_idx].long()
        r_idx = resource[b_idx, h_idx]
        amount = torch.where(state.outpost_upgrade[b_idx, h_idx] == WORKSHOP_INDEX, 2, 1).to(state.resources.dtype)
        state.resources.index_put_((b_idx, f_idx, r_idx), amount, accumulate=True)

    return state


def apply_victory_points(state):
    """End-of-round VP tally (the win condition - see turn.py's
    get_game_winner): a faction's first outpost earns nothing, each
    additional one beyond that earns OUTPOST_VP_PER_ROUND more
    (max(0, outposts - 1), not a flat per-outpost rate), plus a flat
    TEMPLE_VP_PER_ROUND for every outpost with a Temple upgrade. Capitals
    don't count. Destroying an enemy outpost is awarded separately,
    immediately, in turn.py's battle-phase orchestration
    (OUTPOST_DESTROY_VP) - this only covers the recurring per-round
    income."""
    own_outposts = _ownership_mask(state) & (~state.is_capital).unsqueeze(-1)  # [B, N, F]
    outposts = own_outposts.sum(dim=1)  # [B, F]
    is_temple = (state.outpost_upgrade == TEMPLE_INDEX).unsqueeze(-1)  # [B, N, 1]
    temples = (own_outposts & is_temple).sum(dim=1)  # [B, F]
    gain = torch.clamp(outposts - 1, min=0) * OUTPOST_VP_PER_ROUND + temples * TEMPLE_VP_PER_ROUND
    state.victory_points += gain.to(state.victory_points.dtype)
    return state


def apply_collect_phase(state, decide_resource_choice_list):
    """Runs the full Collect phase in place: gold income, resource
    income, then the per-round VP tally (each independently testable
    above - see module docstring for why they're bundled into one call
    from turn.py)."""
    apply_gold_income(state)
    apply_resource_income(state, decide_resource_choice_list)
    apply_victory_points(state)
    return state
