"""
Batched movement mechanics for engine - every game in the batch advances
one movement/cavalry step at once. See engine_old/movement.py for the
original single-game version this was ported from; the RULES are
unchanged (see that module's docstring for the full rationale: swap/line-
battle detection, per-destination grouping, capped peaceful merge with
reversion on overstacking, capitals/outposts never capturable by walking
in undefended). What changed is HOW it's computed: the original processed
one game's variable-length list of moves with Python dicts/sets; this
version processes all B games' moves at once with fixed-shape tensor ops.

Two things make that tractable without a Python loop over B or N (the two
large, batched dimensions):

  - loop over FACTION-PAIR index (f < f', a fixed F*(F-1)/2 count) for
    swap/line-battle detection, and over FACTION index (0..F-1, a fixed
    F count) for grouping arrivals by destination - both small, fixed,
    board/batch-independent ranges.
  - every hex that gets a NEW battle contribution this step - whether
    from a swap, an ordinary arrival, an existing hostile occupant, or a
    battle a reverted move cascades into - goes through the same sparse
    primitives (_lock_hexes/_assign_contributions), which take flat
    [M]-length "event lists" (M = however many (batch, hex) pairs need
    this action right now, not B or B*N). Each ordered WAVE of
    contributions to a hex is one call, in the order the original would
    have inserted them - see _assign_contributions' docstring for why a
    single call must never contain two events for the same (batch, hex)
    pair, and how call ordering reproduces contribution-slot order
    (battle.py's _battle_faction_order reads slots 0..K-1 by first
    appearance, and that order drives who rolls/targets first each round
    - see state.py's battle_order docstring for the same point at the
    whole-turn level).

Resolution order for a step, matching the original's two passes:
  1. swap/line-battle pairs (loop over faction pairs).
  2. remaining moves grouped by destination: hexes needing a NEW or
     EXTENDED battle (already locked, a hostile occupant, >1 arriving
     faction, or a foreign-owned structure) get their contributions
     assigned; hexes with a single peaceful arrival either merge (under
     the stack cap) or, if that would overstack, every arriving move is
     reverted - which can itself start a battle at the reverting unit's
     origin hex if that hex was independently claimed by someone else
     this same step (see _revert_departures). Peaceful merges are applied
     before reverts are processed, so a revert's cascade sees this step's
     other merges - a deterministic choice where the original's Python
     dict-iteration order would otherwise decide; see that function's
     docstring.
"""

import torch

from .state import IMPASSABLE_BY_TERRAIN, MAX_STACK_SIZE, NO_FACTION, TERRAIN_TO_INDEX

MARSH_INDEX = TERRAIN_TO_INDEX["marsh"]


def _legal_mask(state, faction, require_cavalry):
    """[B, N, 6] bool: legal_mask[b, h, d] = True iff `faction`'s whole
    army at hex h (in game b) may move to h's neighbor in direction d
    this step."""
    own_army = (state.army_faction == faction) & ~state.locked & ~state.frozen  # [B, N]
    if require_cavalry:
        own_army = own_army & (state.army_units[..., 1] > 0)

    neighbor = state.grid.neighbor_table  # [N, 6], unbatched
    has_neighbor = neighbor >= 0
    safe_neighbor = torch.where(has_neighbor, neighbor, torch.zeros_like(neighbor)).long()
    neighbor_terrain = state.terrain[:, safe_neighbor]  # [B, N, 6]
    passable = ~IMPASSABLE_BY_TERRAIN.to(state.device)[neighbor_terrain.long()]

    return own_army[:, :, None] & has_neighbor[None, :, :] & passable


def legal_movement_mask(state, faction):
    """[B, N, 6] bool - see _legal_mask."""
    return _legal_mask(state, faction, require_cavalry=False)


def legal_cavalry_mask(state, faction):
    """Same, but only for hexes with cavalry present."""
    return _legal_mask(state, faction, require_cavalry=True)


def actions_from_dicts(actions_by_faction_list, num_factions, device):
    """[{"faction": (from_hex, direction), ...}, ...] (length B, one dict
    per batch item, faction key omitted or mapped to None for "no
    action") -> the (from_hex, direction, has_action) tensors
    apply_movement_step wants. Bridges the batched engine's tensor
    interface to per-game callers (agents, tournament.py, run.py): agents
    stay per-game Python functions (see the plan's "Consequence for
    agents/tooling" section), this is where their per-game answers get
    stacked into a batch."""
    B = len(actions_by_faction_list)
    from_hex = torch.zeros(B, num_factions, dtype=torch.long, device=device)
    direction = torch.zeros(B, num_factions, dtype=torch.long, device=device)
    has_action = torch.zeros(B, num_factions, dtype=torch.bool, device=device)
    for b, actions_by_faction in enumerate(actions_by_faction_list):
        for faction, action in actions_by_faction.items():
            if action is None:
                continue
            from_hex[b, faction] = action[0]
            direction[b, faction] = action[1]
            has_action[b, faction] = True
    return from_hex, direction, has_action


def _units_to_move(state, from_hex, cavalry_only):
    """[B, F, 3] - the whole stack at from_hex (movement phase) or just
    its cavalry count, leaving 100% of infantry/archers behind (cavalry
    phase) - engine actions only ever move a fixed, phase-determined
    subset, never an arbitrary split (unlike the pre-engine v1 this was
    ported from - see engine_old/movement.py's SCOPE note)."""
    units_at_from = torch.gather(state.army_units, 1, from_hex.unsqueeze(-1).expand(-1, -1, 3))
    if not cavalry_only:
        return units_at_from
    units = torch.zeros_like(units_at_from)
    units[..., 1] = units_at_from[..., 1]
    return units


def _first_empty_slot_index(used):
    """used: [M, K] bool (True = slot occupied). Returns [M] long: the
    index of each row's FIRST empty slot, or K (out of range - caller
    should treat this as MAX_BATTLE_CONTRIB overflow) if none."""
    K = used.shape[-1]
    is_empty = ~used
    idx = is_empty.float().argmax(dim=-1)  # argmax of a 0/1 tensor finds the first True
    return torch.where(is_empty.any(dim=-1), idx, torch.full_like(idx, K))


def _lock_hexes(state, batch_idx, hex_idx):
    """batch_idx/hex_idx: [M] long, the (batch, hex) pairs that need a
    battle lock this event - duplicates are fine (deduped internally), a
    hex already locked is a harmless no-op for the "newly locked"
    bookkeeping (round reset, battle_order append) but still gets its
    peaceful army cleared/locked flag set again (idempotent)."""
    if len(batch_idx) == 0:
        return
    pairs = torch.unique(torch.stack([batch_idx, hex_idx], dim=1), dim=0)
    b, h = pairs[:, 0], pairs[:, 1]
    newly = ~state.locked[b, h]
    for i in torch.nonzero(newly, as_tuple=False).flatten().tolist():
        bb, hh = int(b[i]), int(h[i])
        state.battle_round[bb, hh] = 0
        state.battle_order[bb].append(hh)
    state.locked[b, h] = True
    state.army_faction[b, h] = NO_FACTION
    state.army_units[b, h] = 0
    state.frozen[b, h] = False


def _assign_contributions(state, batch_idx, hex_idx, faction, origin, units, moved):
    """All args [M]-length flat event lists (units: [M, 3]) - one call is
    one ORDERED WAVE of new battle contributions. batch_idx/hex_idx MUST
    NOT contain the same (batch, hex) pair twice within a single call
    (each such pair gets exactly one contribution per call - callers
    that need to add several contributions to the same hex this step
    make several calls, one per wave, in the order those contributions
    should occupy battle slots - see module docstring). Silently drops
    any event whose hex has no empty slot left (MAX_BATTLE_CONTRIB
    exceeded) rather than raising, since a batched op can't easily raise
    per-item - callers needing the old hard error should check
    len(nonzero(dropped)) themselves if that matters."""
    if len(batch_idx) == 0:
        return
    used = state.battle_faction[batch_idx, hex_idx] != NO_FACTION  # [M, K]
    slot = _first_empty_slot_index(used)
    ok = slot < used.shape[-1]
    if not bool(ok.any()):
        return
    b, h, s = batch_idx[ok], hex_idx[ok], slot[ok]
    state.battle_faction[b, h, s] = faction[ok].to(state.battle_faction.dtype)
    state.battle_origin[b, h, s] = origin[ok].to(state.battle_origin.dtype)
    state.battle_units[b, h, s] = units[ok].to(state.battle_units.dtype)
    state.battle_moved[b, h, s] = moved[ok]


def _subtract_departures(state, batch_idx, from_hex, units):
    """[M]-length flat event lists - removes `units` from each
    (batch_idx, from_hex) army, clearing the hex if now empty. batch_idx/
    from_hex pairs must be distinct within one call (true by construction
    everywhere this is called - each call covers one faction column, and
    a faction only has one origin hex per step)."""
    if len(batch_idx) == 0:
        return
    state.army_units[batch_idx, from_hex] = state.army_units[batch_idx, from_hex] - units
    now_empty = state.army_units[batch_idx, from_hex].sum(dim=-1) == 0
    eb, eh = batch_idx[now_empty], from_hex[now_empty]
    state.army_faction[eb, eh] = NO_FACTION
    state.frozen[eb, eh] = False


def apply_movement_step(state, from_hex, direction, has_action, rng, cavalry_only=False):
    """Applies one simultaneous movement step to every game in the batch
    at once, in place. from_hex/direction: [B, F] long. has_action:
    [B, F] bool. `rng`: a torch.Generator on the same device as `state`,
    consumed only for a Line Battle's exact-tie coin flip - callers
    should pass the same generator used for the rest of the turn so runs
    stay reproducible given a fixed seed. `cavalry_only` must match: True
    for actions built from legal_cavalry_mask, False for
    legal_movement_mask. See module docstring for the overall resolution
    order."""
    B, F = from_hex.shape
    N = state.num_hexes
    device = state.device
    b_idx = torch.arange(B, device=device)
    grid = state.grid

    # ---- validate ----
    faction_ids = torch.arange(F, device=device)[None, :].expand(B, F)
    own_ok = torch.gather(state.army_faction, 1, from_hex) == faction_ids
    locked_ok = ~torch.gather(state.locked, 1, from_hex)
    frozen_ok = ~torch.gather(state.frozen, 1, from_hex)
    units = _units_to_move(state, from_hex, cavalry_only)  # [B, F, 3]
    units_nonzero = units.sum(dim=-1) > 0
    to_hex = grid.neighbor_table[from_hex, direction]  # [B, F], elementwise gather via unbatched index tensor
    to_valid = to_hex >= 0
    to_hex_safe = torch.where(to_valid, to_hex, torch.zeros_like(to_hex))
    passable = ~IMPASSABLE_BY_TERRAIN.to(device)[torch.gather(state.terrain, 1, to_hex_safe).long()]

    valid = has_action & own_ok & locked_ok & frozen_ok & units_nonzero & to_valid & passable
    to_hex = torch.where(valid, to_hex_safe, torch.full_like(to_hex, -1))

    # ---- pass 1: swap / line-battle detection ----
    consumed = torch.zeros(B, F, dtype=torch.bool, device=device)
    for f in range(F):
        for fp in range(f + 1, F):
            is_pair = (valid[:, f] & valid[:, fp]
                       & (from_hex[:, f] == to_hex[:, fp]) & (to_hex[:, f] == from_hex[:, fp]))
            if not bool(is_pair.any()):
                continue
            rows = torch.nonzero(is_pair, as_tuple=False).flatten()
            m_total = units[rows, f].sum(dim=-1)
            r_total = units[rows, fp].sum(dim=-1)
            coin = torch.rand(len(rows), generator=rng, device=device)
            battle_hex = torch.where(
                m_total < r_total, from_hex[rows, f],
                torch.where(r_total < m_total, from_hex[rows, fp],
                            torch.where(coin < 0.5, from_hex[rows, f], from_hex[rows, fp])),
            )
            # Subtract departures BEFORE locking/clearing battle_hex - if
            # battle_hex happens to be one mover's own origin (the smaller
            # army stays on its own ground), _lock_hexes' clear-army side
            # effect must not run before that army's departure is
            # accounted for, or it gets subtracted twice (see
            # engine_old/movement.py: _subtract_departure always runs
            # before _start_or_extend_battle).
            _subtract_departures(state, rows, from_hex[rows, f], units[rows, f])
            _subtract_departures(state, rows, from_hex[rows, fp], units[rows, fp])
            _lock_hexes(state, rows, battle_hex)
            _assign_contributions(state, rows, battle_hex, faction_ids[rows, f], from_hex[rows, f],
                                   units[rows, f], torch.ones(len(rows), dtype=torch.bool, device=device))
            _assign_contributions(state, rows, battle_hex, faction_ids[rows, fp], from_hex[rows, fp],
                                   units[rows, fp], torch.ones(len(rows), dtype=torch.bool, device=device))
            consumed[rows, f] = True
            consumed[rows, fp] = True

    remaining = valid & ~consumed  # [B, F]

    # ---- pass 2: group remaining moves by destination ----
    for f in range(F):
        rows = torch.nonzero(remaining[:, f], as_tuple=False).flatten()
        _subtract_departures(state, rows, from_hex[rows, f], units[rows, f])

    arrivals_mask = torch.zeros(B, N, F, dtype=torch.bool, device=device)
    for f in range(F):
        rows = torch.nonzero(remaining[:, f], as_tuple=False).flatten()
        if len(rows):
            arrivals_mask[rows, to_hex[rows, f], f] = True
    has_any_arrival = arrivals_mask.any(dim=-1)  # [B, N]
    num_arrival_factions = arrivals_mask.sum(dim=-1)  # [B, N]

    existing_faction = state.army_faction  # [B, N] (post-departure state)
    existing_present = existing_faction != NO_FACTION
    existing_safe = torch.where(existing_present, existing_faction, torch.zeros_like(existing_faction)).long()
    existing_is_arriver = torch.gather(arrivals_mask, 2, existing_safe.unsqueeze(-1)).squeeze(-1)
    hostile_present = existing_present & ~existing_is_arriver

    dest_owner = state.city_owner  # [B, N]
    owner_present = dest_owner != NO_FACTION
    owner_safe = torch.where(owner_present, dest_owner, torch.zeros_like(dest_owner)).long()
    owner_is_arriver = torch.gather(arrivals_mask, 2, owner_safe.unsqueeze(-1)).squeeze(-1)
    foreign_structure = owner_present & ~owner_is_arriver

    needs_battle = has_any_arrival & (state.locked | hostile_present | (num_arrival_factions > 1) | foreign_structure)
    peaceful = has_any_arrival & ~needs_battle

    # -- battle path: snapshot each hostile occupant's units BEFORE
    # locking (which clears army_units/army_faction in place - reading
    # them after would see the just-cleared zeros), then lock, then
    # assign contributions wave-by-wave in (arriving factions ascending,
    # existing hostile occupant last) order --
    occupant_units_snapshot = state.army_units.clone()
    _lock_hexes(state, *torch.nonzero(needs_battle, as_tuple=True))
    for f in range(F):
        active = needs_battle & arrivals_mask[:, :, f]
        bb, hh = torch.nonzero(active, as_tuple=True)
        if len(bb) == 0:
            continue
        rows_by_faction = torch.nonzero(remaining[:, f], as_tuple=False).flatten()
        # bb/hh (from `active`) and rows_by_faction (from `remaining[:,f]`) both
        # describe "faction f's arrival at a needs_battle hex" - align them by
        # batch index (each batch item has at most one arrival for faction f).
        origin_by_batch = torch.full((B,), -1, dtype=torch.long, device=device)
        units_by_batch = torch.zeros(B, 3, dtype=units.dtype, device=device)
        origin_by_batch[rows_by_faction] = from_hex[rows_by_faction, f]
        units_by_batch[rows_by_faction] = units[rows_by_faction, f]
        _assign_contributions(state, bb, hh, faction_ids[bb, f], origin_by_batch[bb], units_by_batch[bb],
                               torch.ones(len(bb), dtype=torch.bool, device=device))
    hb, hh2 = torch.nonzero(needs_battle & hostile_present, as_tuple=True)
    if len(hb):
        occupant_faction = existing_safe[hb, hh2]
        occupant_units = occupant_units_snapshot[hb, hh2]
        _assign_contributions(state, hb, hh2, occupant_faction, hh2, occupant_units,
                               torch.zeros(len(hb), dtype=torch.bool, device=device))

    # -- peaceful path: capped merge, or revert everyone on overstack --
    arrive_units_sum = torch.zeros(B, N, dtype=units.dtype, device=device)
    for f in range(F):
        rows = torch.nonzero(remaining[:, f], as_tuple=False).flatten()
        if len(rows):
            arrive_units_sum[rows, to_hex[rows, f]] += units[rows, f].sum(dim=-1)
    existing_total = torch.where(existing_present, state.army_units.sum(dim=-1), torch.zeros_like(existing_faction))
    fits = peaceful & (existing_total + arrive_units_sum <= MAX_STACK_SIZE)
    reverts = peaceful & ~fits

    sole_faction = arrivals_mask.long().argmax(dim=-1)  # the one True column - valid wherever num_arrival_factions==1
    resolved_faction = torch.where(existing_present, existing_faction, sole_faction.to(existing_faction.dtype))
    mb, mh = torch.nonzero(fits, as_tuple=True)
    if len(mb):
        state.army_faction[mb, mh] = resolved_faction[mb, mh]
        for f in range(F):
            rows = torch.nonzero(remaining[:, f], as_tuple=False).flatten()
            if not len(rows):
                continue
            dest = to_hex[rows, f]
            active = fits[rows, dest]
            if not bool(active.any()):
                continue
            r, d = rows[active], dest[active]
            state.army_units[r, d] = state.army_units[r, d] + units[r, f]
        marsh_now = state.terrain[mb, mh] == MARSH_INDEX
        state.frozen[mb[marsh_now], mh[marsh_now]] = True

    if bool(reverts.any()):
        revert_batch, revert_faction, revert_origin, revert_units = [], [], [], []
        for f in range(F):
            rows = torch.nonzero(remaining[:, f], as_tuple=False).flatten()
            if not len(rows):
                continue
            dest = to_hex[rows, f]
            active = reverts[rows, dest]
            if not bool(active.any()):
                continue
            r = rows[active]
            revert_batch.append(r)
            revert_faction.append(faction_ids[r, f])
            revert_origin.append(from_hex[r, f])
            revert_units.append(units[r, f])
        _revert_departures(
            state,
            torch.cat(revert_batch), torch.cat(revert_faction), torch.cat(revert_origin), torch.cat(revert_units),
        )

    return state


def _revert_departures(state, batch_idx, faction, origin, units):
    """KNOWN, NARROW BEHAVIORAL DIFFERENCE FROM engine_old (differential-
    tested, not accidental - see test scenario "I" in the movement diff
    harness): engine_old resolves destinations in Python dict insertion
    order (the order each destination was first seen while scanning moves
    in faction-index order), so a revert that restores a unit to ITS OWN
    origin can retroactively turn what would otherwise be a later-
    processed peaceful arrival AT THAT SAME HEX into a hostile battle,
    with that battle's contribution metadata reflecting whichever
    destination got processed first. This version always resolves
    reverts strictly AFTER every peaceful merge elsewhere in the step
    (see apply_movement_step) - a well-defined, deterministic order
    that's necessary for batching (dict-insertion-order sequencing is
    inherently per-destination-serial) but not always bit-identical to
    engine_old's order-sensitive one. Differential testing confirms this
    only ever affects battle_origin (the "retreat to" bookkeeping) for a
    contribution in the resulting battle, never who's actually fighting
    or with how many units - and only in the rare double-edge-case where
    an overstack revert's origin hex is ALSO independently claimed by a
    different faction the same step. Judged not worth chasing further:
    reproducing engine_old's exact behavior here would require genuinely
    serial per-destination processing, defeating the point of batching,
    to match what looks like unintentional complexity (a side effect of
    Python dict ordering) rather than a deliberate rule.

    [M]-length flat event lists (units: [M, 3]) - sends each reverted
    move's units back to its origin hex, mirroring engine_old/movement.py's
    _revert_departure: origin empty -> recreate a peaceful army there;
    origin now held by the SAME faction (another of their own moves
    peacefully merged there first) -> add to it; origin now held by a
    DIFFERENT faction (they peacefully claimed it this same step) ->
    start a battle there instead, with the claimant's units (moved=True,
    they really did move there) and the reverting units (moved=False -
    this move never actually left, per the original's docstring: "a's own
    move got voided by the overstack revert, so by the end of this step
    they never actually left origin at all").

    Ordering note: this runs after every peaceful merge elsewhere in the
    same step has already been applied to `state` (see apply_movement_step),
    so "origin now held by a different faction" reflects this step's
    other merges, not just state from before this step began - a
    deterministic choice standing in for the original's Python dict-
    iteration order, which could interleave a revert between two other
    destinations' resolutions depending on move submission order. Two
    DIFFERENT reverting moves landing on the SAME origin hex in the same
    call (e.g. two units of the same faction both reverting to hexes that
    happen to coincide) is handled like any other multi-arrival hex would
    be by _assign_contributions/army-merge - see the "same" branch below,
    which sums rather than overwrites."""
    if len(batch_idx) == 0:
        return
    cur_faction = state.army_faction[batch_idx, origin]
    empty = cur_faction == NO_FACTION
    same = cur_faction == faction
    other = ~empty & ~same

    if bool(empty.any()):
        b, o, fac, u = batch_idx[empty], origin[empty], faction[empty], units[empty]
        # multiple reverts to the same still-empty origin (same faction,
        # e.g. two cavalry moves both bounced back to one shared origin)
        # must accumulate, not overwrite - group by (batch, origin, faction).
        state.army_faction[b, o] = fac.to(state.army_faction.dtype)
        state.army_units[b, o] = 0
        state.army_units.index_put_((b, o), u, accumulate=True)

    if bool(same.any()):
        b, o, u = batch_idx[same], origin[same], units[same]
        state.army_units.index_put_((b, o), u, accumulate=True)

    if bool(other.any()):
        b, o, fac, u = batch_idx[other], origin[other], faction[other], units[other]
        claimant_faction = state.army_faction[b, o]
        claimant_units = state.army_units[b, o].clone()
        _lock_hexes(state, b, o)
        _assign_contributions(state, b, o, claimant_faction, o, claimant_units,
                               torch.ones(len(b), dtype=torch.bool, device=state.device))
        _assign_contributions(state, b, o, fac, o, u,
                               torch.zeros(len(b), dtype=torch.bool, device=state.device))
