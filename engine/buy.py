"""
Batched Buy phase for engine.

Four kinds of purchases, atomic (one unit each): buy_infantry (spend 2
gold at an owned city - capital or outpost), convert_to_special (spend 1
kill-XP + 1 gold to convert an existing infantry unit into cavalry/
archers), build_outpost (spend 3 gold + consume 1 unit already standing
on the target hex to found a new outpost), and upgrade_outpost (spend
that upgrade's resource cost to give an owned outpost a Barracks/
Workshop/Temple). buy_infantry/convert_to_special are subject to the
24/12/12 concurrent SPAWN_CAPS; build_outpost is subject to OUTPOST_CAP
instead plus eligible_outpost_mask's placement-distance rules; outpost
actions (build+upgrade combined) and buy_infantry AT AN OUTPOST (not a
capital) are each capped at 1/turn - see apply_buy_phase_batch.

FIXED-SHAPE ACTION REDESIGN (the milestone engine_old/buy.py's docstring
flagged as deferred: "the fixed/masked action-space design discussed for
the eventual RL policy"). A buy decision, per (batch item, faction), is
no longer a variable-length list of action dicts - it's a fixed
structure:

    outpost_type: 0 (none) / 1 (build) / 2 (upgrade)
    outpost_hex: the target hex (-1 if outpost_type is 0)
    outpost_unit_type: which unit build_outpost consumes (0/1/2)
    outpost_upgrade: which upgrade upgrade_outpost applies (0/1/2)
    infantry_buy: [N] int - desired infantry purchases AT EACH hex
    convert_cavalry: [N] int - desired infantry->cavalry conversions AT EACH hex
    convert_archers: [N] int - desired infantry->archers conversions AT EACH hex

This is strictly more general than any current agent's actual behavior
(greedy_buy, for instance, only ever wants infantry at ONE hex, but
nothing here requires that) while being fully fixed-shape - a per-hex
vector, not a list. It's fulfilled via a bounded SHOT-LOOP (same pattern
used throughout movement.py/battle.py for "resolve up to K sequential,
state-dependent events, batched across items"): each iteration finds the
lowest-index hex with remaining demand for every (batch, faction) at
once, attempts that one purchase (checking legality/affordability/caps
fresh, since an earlier iteration this same phase may have changed
them), and moves on - correctly reproducing the ORIGINAL per-purchase
sequential legality/affordability checks (gold and SPAWN_CAPS are shared,
mutable resources purchases compete for) without a Python loop over
either the batch or the board.

Processing order within one buy phase, matching engine_old's own
ordering (agents built their action list outpost-action-first, then
infantry, then conversions - and since gold is shared across all three,
order affects outcomes when gold is the binding constraint): outpost
action, then infantry purchases, then conversions.
"""

import torch

from .geometry import min_hex_distance_to_any
from .state import NO_FACTION, RESOURCE_TO_INDEX, SPAWN_CAPS, UPGRADE_TO_INDEX

INFANTRY_COST = 2
CAVALRY = 1
ARCHERS = 2

OUTPOST_COST = 3
OUTPOST_CAP = 6
OUTPOST_MIN_DIST_OWN_CAPITAL = 3    # "not within 2 tiles of your own capital"
OUTPOST_MIN_DIST_ENEMY_CAPITAL = 2  # "not within 1 tile of any other faction's capital"
OUTPOST_MIN_DIST_OTHER_OUTPOST = 2  # "not within 1 tile of any outpost" (yours or anyone else's)
UNIT_TYPE_INDEX = {"infantry": 0, "cavalry": 1, "archers": 2}

BARRACKS_INDEX = UPGRADE_TO_INDEX["barracks"]

# Resource cost to give an outpost each upgrade (or to convert it directly
# from a different upgrade - full cost, no credit for the one replaced).
# Index-keyed (0=barracks, 1=workshop, 2=temple - see state.UPGRADE_TYPES)
# for the batched engine; RESOURCE_COST_MATRIX below is the same data as
# one [3, 4] tensor for vectorized affordability checks.
UPGRADE_COSTS = {
    UPGRADE_TO_INDEX["barracks"]: {"fish": 2, "wood": 4},
    UPGRADE_TO_INDEX["workshop"]: {"iron": 2, "clay": 2, "wood": 4},
    UPGRADE_TO_INDEX["temple"]: {"fish": 2, "iron": 2, "clay": 2, "wood": 4},
}


def _make_cost_matrix(device):
    m = torch.zeros(3, 4, dtype=torch.long, device=device)
    for upgrade_index, cost in UPGRADE_COSTS.items():
        for r, amount in cost.items():
            m[upgrade_index, RESOURCE_TO_INDEX[r]] = amount
    return m


def _outpost_count(state, faction):
    """[B] int - how many outposts (non-capital owned cities) `faction`
    currently holds."""
    return ((state.city_owner == faction) & ~state.is_capital).sum(dim=-1)


def eligible_outpost_mask(state, faction):
    """[B, N] bool - the same legality rules as before the rewrite (not on
    top of an existing capital/outpost, >= OUTPOST_MIN_DIST_OWN_CAPITAL
    from your own capital, >= OUTPOST_MIN_DIST_ENEMY_CAPITAL from any
    other faction's capital, >= OUTPOST_MIN_DIST_OTHER_OUTPOST from any
    outpost at all), now for every (batch item, hex) at once. Distance
    checks are still per-batch-item (each game's capitals/outposts sit at
    different hexes), so this loops over the batch for the distance parts
    - cheap, since it's called once per buy-phase decision, not once per
    candidate hex like the callers this was originally optimized for."""
    B, N = state.terrain.shape
    device = state.device
    coords = state.grid.coords_array
    mask = state.city_owner == NO_FACTION  # [B, N]

    for b in range(B):
        own_capital = torch.nonzero((state.city_owner[b] == faction) & state.is_capital[b], as_tuple=False).flatten()
        if len(own_capital):
            dist = min_hex_distance_to_any(coords, own_capital)
            mask[b] &= dist >= OUTPOST_MIN_DIST_OWN_CAPITAL

        enemy_capitals = torch.nonzero(
            state.is_capital[b] & (state.city_owner[b] != NO_FACTION) & (state.city_owner[b] != faction),
            as_tuple=False,
        ).flatten()
        if len(enemy_capitals):
            dist = min_hex_distance_to_any(coords, enemy_capitals)
            mask[b] &= dist >= OUTPOST_MIN_DIST_ENEMY_CAPITAL

        all_outposts = torch.nonzero(
            (state.city_owner[b] != NO_FACTION) & ~state.is_capital[b], as_tuple=False
        ).flatten()
        if len(all_outposts):
            dist = min_hex_distance_to_any(coords, all_outposts)
            mask[b] &= dist >= OUTPOST_MIN_DIST_OTHER_OUTPOST

    return mask


def _adjacent_enemy_present_batch(state, faction):
    """[B, N] bool - whether an enemy army is adjacent to each hex, for
    `faction` (the siege rule: buy_infantry at an OUTPOST, not a capital,
    requires no adjacent enemy army)."""
    neighbor = state.grid.neighbor_table  # [N, 6]
    valid = neighbor >= 0
    safe = torch.where(valid, neighbor, torch.zeros_like(neighbor)).long()
    neighbor_faction = state.army_faction[:, safe]  # [B, N, 6]
    enemy = valid[None, :, :] & (neighbor_faction != NO_FACTION) & (neighbor_faction != faction)
    return enemy.any(dim=-1)


def _count_all_units_in_play_batch(state):
    """[B, F, 3] int - how many of each faction's infantry/cavalry/
    archers currently exist, on the board or mid-battle. einsum operands
    are cast to float32 (not int16/int8) - CUDA's einsum kernels don't
    cover the narrower integer dtypes state.py otherwise uses (hit via
    direct GPU testing, not a theoretical concern); counts here are
    always small enough for float32 to represent exactly."""
    F = state.num_factions
    faction_ids = torch.arange(F, device=state.device)
    board_mask = (state.army_faction.unsqueeze(-1) == faction_ids).float()  # [B, N, F]
    board_total = torch.einsum("bnf,bnu->bfu", board_mask, state.army_units.float())
    battle_mask = (state.battle_faction.unsqueeze(-1) == faction_ids).float()  # [B, N, K, F]
    battle_total = torch.einsum("bnkf,bnku->bfu", battle_mask, state.battle_units.float())
    return (board_total + battle_total).round().long()


def apply_buy_phase_batch(state, outpost_type, outpost_hex, outpost_unit_type, outpost_upgrade,
                           infantry_buy, convert_cavalry, convert_archers):
    """Applies one buy phase to every game in the batch at once, in place.
    outpost_type/outpost_hex/outpost_unit_type/outpost_upgrade: [B, F]
    long. infantry_buy/convert_cavalry/convert_archers: [B, F, N] long -
    desired counts per hex, per faction (see module docstring for why a
    per-hex vector rather than a list). Processing order: outpost action,
    then infantry, then conversions (matches engine_old's own ordering -
    see module docstring)."""
    B, F = outpost_type.shape
    N = state.num_hexes
    device = state.device
    b_idx = torch.arange(B, device=device)
    f_idx = torch.arange(F, device=device)
    cost_matrix = _make_cost_matrix(device)

    counts = _count_all_units_in_play_batch(state)  # [B, F, 3] - running tally, updated as purchases apply
    recruited_this_turn = torch.zeros(B, N, dtype=torch.bool, device=device)  # per-hex, shared across factions
    # (a hex only ever belongs to one faction, so per-hex is equivalent to
    # per-(faction,hex) here - matches engine_old's single outpost_recruited set)

    # ---- outpost action: one per faction, vectorized across (B, F) ----
    for f in range(F):
        want_build = outpost_type[:, f] == 1
        want_upgrade = outpost_type[:, f] == 2
        hex_ = outpost_hex[:, f].clamp(min=0)

        if bool(want_build.any()):
            unit_type = outpost_unit_type[:, f].clamp(min=0, max=2)
            owns_army = state.army_faction[b_idx, hex_] == f
            not_locked = ~state.locked[b_idx, hex_]
            has_unit = state.army_units[b_idx, hex_, unit_type] > 0
            has_gold = state.gold[:, f] >= OUTPOST_COST
            under_cap = _outpost_count(state, f) < OUTPOST_CAP
            eligible = eligible_outpost_mask(state, f)[b_idx, hex_]
            ok = want_build & owns_army & not_locked & has_unit & has_gold & under_cap & eligible
            if bool(ok.any()):
                ob = b_idx[ok]
                oh = hex_[ok]
                out = unit_type[ok]
                state.gold[ob, f] -= OUTPOST_COST
                state.army_units[ob, oh, out] -= 1
                counts[ob, f, out] -= 1
                now_empty = state.army_units[ob, oh].sum(dim=-1) == 0
                state.army_faction[ob[now_empty], oh[now_empty]] = NO_FACTION
                state.city_owner[ob, oh] = f

        if bool(want_upgrade.any()):
            upgrade = outpost_upgrade[:, f].clamp(min=0, max=2)
            owns_city = state.city_owner[b_idx, hex_] == f
            not_capital = ~state.is_capital[b_idx, hex_]
            not_locked = ~state.locked[b_idx, hex_]
            different = state.outpost_upgrade[b_idx, hex_] != upgrade
            cost = cost_matrix[upgrade]  # [B, 4]
            affordable = (state.resources[:, f] >= cost).all(dim=-1)
            ok = want_upgrade & owns_city & not_capital & not_locked & different & affordable
            if bool(ok.any()):
                ub = b_idx[ok]
                uh = hex_[ok]
                uu = upgrade[ok]
                state.resources[ub, f] -= cost[ok]
                state.outpost_upgrade[ub, uh] = uu.to(state.outpost_upgrade.dtype)

    # ---- infantry purchases: shot loop, bounded by the infantry SPAWN_CAP ----
    remaining = infantry_buy.clone()
    max_iters = min(int(SPAWN_CAPS[0]), int(remaining.sum(dim=-1).max()) if remaining.numel() else 0)
    for _ in range(max_iters):
        has_demand = remaining > 0  # [B, F, N]
        has_any = has_demand.any(dim=-1)  # [B, F]
        if not bool(has_any.any()):
            break
        hex_ = has_demand.long().argmax(dim=-1)  # [B, F]

        bf_b = b_idx[:, None].expand(B, F)
        bf_f = f_idx[None, :].expand(B, F)
        owns_city = state.city_owner[bf_b, hex_] == bf_f
        not_locked = ~state.locked[bf_b, hex_]
        is_capital = state.is_capital[bf_b, hex_]
        # Recomputed every iteration (not hoisted above the loop) because
        # an earlier iteration this same phase could have populated a
        # previously-empty hex, changing adjacency for its neighbors - a
        # narrow edge case, but recomputing is the only way to stay
        # correct for it. Costs an extra O(F) pass over the board per
        # iteration; fine at buy-phase's call frequency (once per turn,
        # unlike movement/battle's much hotter per-round loops).
        enemy_adjacent = torch.stack([_adjacent_enemy_present_batch(state, f) for f in range(F)], dim=1)  # [B, F, N]
        no_siege = is_capital | ~enemy_adjacent[bf_b, bf_f, hex_]
        not_recruited_yet = is_capital | (state.outpost_upgrade[bf_b, hex_] == BARRACKS_INDEX) | ~recruited_this_turn[bf_b, hex_]
        has_gold = state.gold >= INFANTRY_COST
        under_cap = counts[..., 0] < int(SPAWN_CAPS[0])
        army_here = state.army_faction[bf_b, hex_]
        occupant_ok = (army_here == NO_FACTION) | (army_here == bf_f)
        room = state.army_units[bf_b, hex_].sum(dim=-1) < 6

        ok = has_any & owns_city & not_locked & no_siege & not_recruited_yet & has_gold & under_cap & occupant_ok & room

        # Demand at THIS hex that turned out illegal this pass is dropped
        # for THIS hex only (matches engine_old: an illegal proposed
        # action is silently skipped, not retried) - other hexes the same
        # faction still wants must stay pending for a later iteration, so
        # this must not clear the whole per-faction demand vector.
        blocked = has_any & ~ok
        if bool(blocked.any()):
            bb, bff = torch.nonzero(blocked, as_tuple=True)
            remaining[bb, bff, hex_[bb, bff]] = 0
        if not bool(ok.any()):
            continue

        ob, of = torch.nonzero(ok, as_tuple=True)
        oh = hex_[ob, of]
        state.gold[ob, of] -= INFANTRY_COST
        newly_owned = state.army_faction[ob, oh] == NO_FACTION
        state.army_faction[ob[newly_owned], oh[newly_owned]] = of[newly_owned].to(state.army_faction.dtype)
        state.army_units[ob, oh, 0] += 1
        counts[ob, of, 0] += 1
        not_barracks_outpost = ~(state.outpost_upgrade[ob, oh] == BARRACKS_INDEX) & ~state.is_capital[ob, oh]
        recruited_this_turn[ob[not_barracks_outpost], oh[not_barracks_outpost]] = True
        remaining[ob, of, oh] -= 1

    # ---- conversions: two shot loops (cavalry, then archers), sharing
    # the same gold/kill_xp pool each attempt draws from ----
    for convert_demand, unit_index in ((convert_cavalry, CAVALRY), (convert_archers, ARCHERS)):
        remaining = convert_demand.clone()
        max_iters = min(int(SPAWN_CAPS[unit_index]), int(remaining.sum(dim=-1).max()) if remaining.numel() else 0)
        for _ in range(max_iters):
            has_demand = remaining > 0
            has_any = has_demand.any(dim=-1)
            if not bool(has_any.any()):
                break
            hex_ = has_demand.long().argmax(dim=-1)

            bf_b = b_idx[:, None].expand(B, F)
            bf_f = f_idx[None, :].expand(B, F)
            owns_army = state.army_faction[bf_b, hex_] == bf_f
            has_infantry = state.army_units[bf_b, hex_, 0] > 0
            has_kill_xp = state.kill_xp >= 1
            has_gold = state.gold >= 1
            under_cap = counts[..., unit_index] < int(SPAWN_CAPS[unit_index])

            ok = has_any & owns_army & has_infantry & has_kill_xp & has_gold & under_cap

            blocked = has_any & ~ok
            if bool(blocked.any()):
                bb, bff = torch.nonzero(blocked, as_tuple=True)
                remaining[bb, bff, hex_[bb, bff]] = 0
            if not bool(ok.any()):
                continue

            ob, of = torch.nonzero(ok, as_tuple=True)
            oh = hex_[ob, of]
            state.kill_xp[ob, of] -= 1
            state.gold[ob, of] -= 1
            state.army_units[ob, oh, 0] -= 1
            state.army_units[ob, oh, unit_index] += 1
            counts[ob, of, 0] -= 1
            counts[ob, of, unit_index] += 1
            remaining[ob, of, oh] -= 1

    return state
