"""
Batched battle resolution for engine.

See engine_old/battle.py for the original single-game version and the
full rules rationale (archer phase, per-round targeting/rolls/kills,
cavalry dismounts, rectification). RULES are unchanged; two things about
HOW they're computed changed for batching:

  - "Battle-slot-index" looping (see engine/turn.py's _run_battle_phase,
    which drives this module): everything here operates on a FLAT,
    SPARSE list of M active (batch, hex) pairs - "whichever games
    currently have a battle pending in this slot" - not on the full
    [B, N, ...] board. M is usually small (at most one hex per game per
    slot) and shrinks as battles resolve, so this stays cheap regardless
    of board size. Turn-level ordering (why a game's battles must resolve
    in a specific sequence, not all at once) is turn.py's concern, not
    this module's - see that module's docstring.
  - Faction order: the original iterates factions in "first battle-slot
    appearance" order, because that order feeds a SINGLE SHARED rng
    stream, so it affects roll sequencing and therefore results
    bit-for-bit. This module targets torch.Generator instead of Python's
    random.Random (see the plan's RNG section) - the two were never going
    to produce bit-identical rolls regardless of ordering - so every
    faction-ordered operation here just uses ascending faction index
    (0..F-1) instead, a simpler, equally-arbitrary, well-defined
    convention. Round/target/roll OUTCOMES can differ from engine_old's
    for this reason; the RULES they're computed under do not.

resolve_full_battle_batch's round loop is a FIXED MAX_ROUNDS_SAFETY_CAP-
iteration masked loop (mirroring the original's `while` with the same
hard cap) instead of a variable-trip-count while loop - a battle that
finishes early just stops contributing further kills/rolls for the rest
of the fixed iteration count (masked out), matching engine_old's actual
behavior (which also just checks is_battle_over each iteration) without
a data-dependent Python loop count.
"""

import torch

from .state import MAX_STACK_SIZE, NO_FACTION, NO_ORIGIN, SPAWN_CAPS, UNIT_TYPES, unstack_states

DEATH_PRIORITY = (0, 1, 2)  # infantry, cavalry, archers
MAX_ROUNDS_SAFETY_CAP = 50  # pure infinite-loop guard - real battles resolve in a handful of rounds
CAPITAL_DEFENSE_SHOTS = 2
OUTPOST_DEFENSE_SHOTS = 1


def faction_totals_sparse(battle_faction, battle_units, num_factions):
    """battle_faction: [M, K], battle_units: [M, K, 3] -> [M, F, 3] int -
    total units per faction, summed over contribution slots (slot ORDER
    doesn't matter here - see module docstring for why "first appearance"
    order isn't reproduced). einsum operands are float32, not int16 -
    CUDA's einsum kernels don't cover the narrower integer dtypes state.py
    otherwise uses (hit via direct GPU testing); counts here are always
    small enough for float32 to represent exactly."""
    F = num_factions
    onehot = (battle_faction.unsqueeze(-1) == torch.arange(F, device=battle_faction.device)).float()
    return torch.einsum("mkf,mku->mfu", onehot, battle_units.float()).round().to(battle_units.dtype)


def faction_moved_totals_sparse(battle_faction, battle_units, battle_moved, num_factions):
    """Like faction_totals_sparse but only over slots with battle_moved
    True - gates the real Archers ability to units that actually moved to
    join the fight (see state.py's battle_moved docstring)."""
    F = num_factions
    onehot = (battle_faction.unsqueeze(-1) == torch.arange(F, device=battle_faction.device)).float()
    onehot = onehot * battle_moved.unsqueeze(-1).float()
    return torch.einsum("mkf,mku->mfu", onehot, battle_units.float()).round().to(battle_units.dtype)


def _kills_for_roll(roll, attacker_total_units):
    """roll/attacker_total_units: [M] int -> [M] int kills. Vectorized
    version of the same three-tier table (0 on 1-5, 1 on 6-15, 1-2 on
    16-20 depending on whether the attacker has exactly 1 unit)."""
    kills = torch.zeros_like(roll)
    kills = torch.where(roll > 5, torch.ones_like(kills), kills)
    kills = torch.where(roll > 15, torch.where(attacker_total_units == 1, 1, 2), kills)
    return kills


def _apply_kills_to_faction(state, batch_idx, hex_idx, target_faction, num_kills):
    """[M] flat event lists - removes up to num_kills units from each
    (batch, hex)'s target_faction contribution slots (infantry -> cavalry
    -> archers cascade, earliest slot index first - slot index order is
    arbitrary now, same reasoning as module docstring's faction-order
    note, but a FIXED order is still needed so kills apply
    deterministically across slots of the SAME faction at the same hex,
    which is why this iterates k=0..K-1 rather than picking arbitrarily).
    Returns [M, 3] int - how many of each unit type were actually killed
    (needed by callers to award kill-XP and detect cavalry deaths for
    dismount rolls)."""
    M = len(batch_idx)
    K = state.battle_faction.shape[2]
    device = state.device
    remaining = num_kills.clone()
    killed = torch.zeros(M, 3, dtype=torch.long, device=device)
    bf = state.battle_faction[batch_idx, hex_idx]  # [M, K]
    bu = state.battle_units[batch_idx, hex_idx].clone()  # [M, K, 3]
    is_target = bf == target_faction.unsqueeze(-1)  # [M, K]

    for ut in DEATH_PRIORITY:
        if not bool((remaining > 0).any()):
            break
        for k in range(K):
            active = (remaining > 0) & is_target[:, k]
            if not bool(active.any()):
                continue
            available = bu[:, k, ut]
            take = torch.minimum(available, remaining)
            take = torch.where(active, take, torch.zeros_like(take))
            bu[:, k, ut] = bu[:, k, ut] - take
            remaining = remaining - take
            killed[:, ut] += take

    state.battle_units[batch_idx, hex_idx] = bu
    return killed


def _roll_d20(m, rng):
    """[m] int in 1..20, drawn from `rng` (a torch.Generator on the same
    device as the caller's tensors)."""
    return (torch.rand(m, generator=rng, device=rng.device) * 20).long() + 1


def _apply_dismount_rolls(state, batch_idx, hex_idx, faction, cav_died, rng, infantry_counts):
    """[M] flat event lists - cav_died: [M] int, how many of `faction`'s
    cavalry died at (batch,hex) this event; rolls the cavalry dismount
    ability once per death (rulebook: "whenever a cavalry unit dies in
    battle" - >=14 on d20 succeeds, subject to the infantry SPAWN_CAP).
    infantry_counts: [B, F] running tally (mutated in place) - shared
    across every battle resolved this turn, same as engine_old, so
    dismount cap checks reflect the whole turn's dismounts so far, not
    just this battle's."""
    max_died = int(cav_died.max()) if len(cav_died) else 0
    device = state.device
    K = state.battle_faction.shape[2]
    for _ in range(max_died):
        active = cav_died > 0
        if not bool(active.any()):
            break
        idx = torch.nonzero(active, as_tuple=False).flatten()
        b, h, f = batch_idx[idx], hex_idx[idx], faction[idx]
        roll = _roll_d20(len(idx), rng)
        succeeded_roll = roll >= 14
        cur_count = infantry_counts[b, f]
        under_cap = cur_count < int(SPAWN_CAPS[0])
        success = succeeded_roll & under_cap

        if bool(success.any()):
            si = idx[success]
            sb, sh, sf = batch_idx[si], hex_idx[si], faction[si]
            bf = state.battle_faction[sb, sh]  # [m, K]
            is_f = bf == sf.unsqueeze(-1)
            first_slot = is_f.long().argmax(dim=-1)
            state.battle_units[sb, sh, first_slot, 0] += 1
            infantry_counts[sb, sf] += 1

        cav_died = cav_died.clone()
        cav_died[idx] -= 1


def apply_structure_defense_shots(state, batch_idx, hex_idx, rng, infantry_counts, kill_xp_delta):
    """[M] flat event lists - runs once, before round 1: a capital or
    outpost gets free defensive shots against whoever's attacking its
    tile, even undefended (2 shots for a capital, 1 for an outpost, each
    11-20 = 1 kill against the largest attacking rival). Adds to
    kill_xp_delta[batch,faction] in place (caller applies it to
    state.kill_xp once, alongside every other phase's kills, matching
    engine_old's per-entry accumulation but batched)."""
    F = state.num_factions
    owner = state.city_owner[batch_idx, hex_idx]  # [M]
    has_owner = owner != NO_FACTION
    if not bool(has_owner.any()):
        return

    totals = faction_totals_sparse(state.battle_faction[batch_idx, hex_idx], state.battle_units[batch_idx, hex_idx], F)
    alive_units = totals.sum(dim=-1)  # [M, F]
    is_rival = (torch.arange(F, device=state.device)[None, :] != owner.unsqueeze(-1)) & (alive_units > 0)
    rival_units = torch.where(is_rival, alive_units, torch.full_like(alive_units, -1))
    has_rival = (rival_units >= 0).any(dim=-1) & has_owner
    target = rival_units.argmax(dim=-1)  # [M]

    active_idx = torch.nonzero(has_rival, as_tuple=False).flatten()
    if len(active_idx) == 0:
        return
    is_capital = state.is_capital[batch_idx[active_idx], hex_idx[active_idx]]
    shots = torch.where(is_capital, CAPITAL_DEFENSE_SHOTS, OUTPOST_DEFENSE_SHOTS)
    max_shots = int(shots.max())
    kills = torch.zeros(len(active_idx), dtype=torch.long, device=state.device)
    for s in range(max_shots):
        still = shots > s
        m = int(still.sum())
        if m == 0:
            continue
        roll = _roll_d20(len(active_idx), rng)
        hit = still & (roll >= 11)
        kills += hit.long()

    has_kills = kills > 0
    if not bool(has_kills.any()):
        return
    ki = active_idx[has_kills]
    b, h = batch_idx[ki], hex_idx[ki]
    killer = owner[ki].long()
    tgt = target[ki]
    killed = _apply_kills_to_faction(state, b, h, tgt, kills[has_kills])
    kill_xp_delta.index_put_((b, killer), killed.sum(dim=-1).to(kill_xp_delta.dtype), accumulate=True)
    cav_died = killed[:, 1]
    _apply_dismount_rolls(state, b, h, tgt, cav_died, rng, infantry_counts)


def apply_archer_abilities(state, batch_idx, hex_idx, rng, infantry_counts, kill_xp_delta):
    """[M] flat event lists - runs once, before round 1. Only archers that
    actually moved to join this battle get to fire (see
    faction_moved_totals_sparse); targeting still weighs each side's full
    strength, moved or not. Faction order is ascending index (see module
    docstring)."""
    F = state.num_factions
    bf = state.battle_faction[batch_idx, hex_idx]
    bu = state.battle_units[batch_idx, hex_idx]
    bm = state.battle_moved[batch_idx, hex_idx]
    totals = faction_totals_sparse(bf, bu, F)  # [M, F, 3]
    moved_totals = faction_moved_totals_sparse(bf, bu, bm, F)  # [M, F, 3]
    alive_units = totals.sum(dim=-1)  # [M, F]

    for f in range(F):
        archers = moved_totals[:, f, 2]
        acting = archers > 0
        if not bool(acting.any()):
            continue
        is_rival = (torch.arange(F, device=state.device)[None, :] != f) & (alive_units > 0)
        rival_units = torch.where(is_rival, alive_units, torch.full_like(alive_units, -1))
        has_rival = (rival_units >= 0).any(dim=-1) & acting
        if not bool(has_rival.any()):
            continue
        target = rival_units.argmax(dim=-1)

        idx = torch.nonzero(has_rival, as_tuple=False).flatten()
        archer_count = archers[idx]  # [M'] - each entry's fixed shot count for this phase
        max_archers = int(archer_count.max())
        kills = torch.zeros(len(idx), dtype=torch.long, device=state.device)
        for s in range(max_archers):
            still_has_shot = archer_count > s
            roll = _roll_d20(len(idx), rng)
            hit = still_has_shot & (roll >= 11)
            kills += hit.long()

        has_kills = kills > 0
        if not bool(has_kills.any()):
            continue
        ki = idx[has_kills]
        b, h = batch_idx[ki], hex_idx[ki]
        killer = torch.full_like(b, f)
        tgt = target[ki]
        killed = _apply_kills_to_faction(state, b, h, tgt, kills[has_kills])
        kill_xp_delta.index_put_((b, killer), killed.sum(dim=-1).to(kill_xp_delta.dtype), accumulate=True)
        _apply_dismount_rolls(state, b, h, tgt, killed[:, 1], rng, infantry_counts)

        # refresh totals/alive for the next faction's targeting
        bf = state.battle_faction[batch_idx, hex_idx]
        bu = state.battle_units[batch_idx, hex_idx]
        totals = faction_totals_sparse(bf, bu, F)
        alive_units = totals.sum(dim=-1)


def _alive_counts(state, batch_idx, hex_idx):
    """[M, F] int - per-faction total units at each (batch, hex) pair."""
    return faction_totals_sparse(
        state.battle_faction[batch_idx, hex_idx], state.battle_units[batch_idx, hex_idx], state.num_factions
    ).sum(dim=-1)


def is_battle_over_batch(state, batch_idx, hex_idx):
    """[M] bool - True where at most one faction still has any units."""
    alive = _alive_counts(state, batch_idx, hex_idx) > 0  # [M, F]
    return alive.sum(dim=-1) <= 1


def get_winner_batch(state, batch_idx, hex_idx):
    """[M] long - the sole alive faction at each (batch, hex) pair, or -1
    if the battle isn't over (more than one alive) or somehow empty."""
    alive = _alive_counts(state, batch_idx, hex_idx) > 0  # [M, F]
    count = alive.sum(dim=-1)
    winner = torch.where(count == 1, alive.long().argmax(dim=-1), torch.full_like(count, NO_FACTION))
    return winner


def resolve_round_batch(state, batch_idx, hex_idx, target_choice, rng, infantry_counts, kill_xp_delta):
    """One full round, for every (batch, hex) pair in this call at once.
    target_choice: [M, F] long, -1 for "no target"/"not alive" - the
    caller (resolve_full_battle_batch) is responsible for gathering these
    from each active game's per-faction agent callback before calling
    this. See module docstring for why faction order is ascending index,
    not "first battle-slot appearance" like engine_old."""
    M = len(batch_idx)
    F = state.num_factions
    device = state.device

    totals = _alive_counts(state, batch_idx, hex_idx)  # [M, F] - fixed for the whole round (simultaneous resolution)

    # -- conflict resolution: among attackers sharing a target, the one
    # with the most units wins (ties broken toward the lower faction
    # index) - see module docstring for _resolve_targets' vectorized form.
    faction_idx = torch.arange(F, device=device)
    score = totals * (F + 1) - faction_idx[None, :]  # [M, F], strictly higher score always wins
    same_target = (target_choice.unsqueeze(2) == target_choice.unsqueeze(1)) & (target_choice.unsqueeze(2) >= 0)  # [M,F,F]
    masked_score = torch.where(same_target, score.unsqueeze(1).expand(-1, F, -1), torch.full_like(score, -1).unsqueeze(1).expand(-1, F, -1))
    group_max = masked_score.max(dim=-1).values  # [M, F]
    wins = (target_choice >= 0) & (score >= group_max)
    resolved_target = torch.where(wins, target_choice, torch.full_like(target_choice, -1))  # [M, F]

    # -- rolls + kills, one attacking faction column at a time (fixed F
    # loop) so multiple killers hitting the same target this round apply
    # sequentially and correctly accumulate (see _apply_kills_to_faction) --
    cav_died_by_target = torch.zeros(M, F, dtype=torch.long, device=device)
    for f in range(F):
        active = resolved_target[:, f] >= 0
        if not bool(active.any()):
            continue
        idx = torch.nonzero(active, as_tuple=False).flatten()
        roll = _roll_d20(len(idx), rng)
        kills = _kills_for_roll(roll, totals[idx, f])
        has_kills = kills > 0
        if not bool(has_kills.any()):
            continue
        ki = idx[has_kills]
        b, h = batch_idx[ki], hex_idx[ki]
        tgt = resolved_target[ki, f]
        killed = _apply_kills_to_faction(state, b, h, tgt, kills[has_kills])
        killer = torch.full_like(b, f)
        kill_xp_delta.index_put_((b, killer), killed.sum(dim=-1).to(kill_xp_delta.dtype), accumulate=True)
        cav_died_by_target[ki, tgt] += killed[:, 1]

    for t in range(F):
        died = cav_died_by_target[:, t]
        active = died > 0
        if not bool(active.any()):
            continue
        idx = torch.nonzero(active, as_tuple=False).flatten()
        b, h = batch_idx[idx], hex_idx[idx]
        faction_t = torch.full_like(b, t)
        _apply_dismount_rolls(state, b, h, faction_t, died[idx], rng, infantry_counts)

    state.battle_round[batch_idx, hex_idx] += 1


def _gather_target_choices(state, batch_idx, hex_idx, decide_target_list, state_views, alive_mask):
    """alive_mask: [M, F] bool - only calls the per-game callback for
    (m, f) pairs where faction f is currently alive in that battle (a
    faction with no units left has nothing to decide). Returns
    target_choice: [M, F] long, -1 for no-target/not-called. Agents stay
    per-game Python functions (see module docstring), so this is a
    Python loop over the M active battles x up to F factions - the cost
    this design accepts in exchange for keeping agent logic unbatched
    (see the plan's "Consequence for agents/tooling" section)."""
    M, F = alive_mask.shape
    target_choice = torch.full((M, F), -1, dtype=torch.long, device=state.device)
    for m in range(M):
        b, h = int(batch_idx[m]), int(hex_idx[m])
        for f in range(F):
            if not bool(alive_mask[m, f]):
                continue
            choice = decide_target_list[b][f](state_views[b], h, f)
            if choice is not None:
                target_choice[m, f] = choice
    return target_choice


def resolve_full_battle_batch(state, batch_idx, hex_idx, decide_target_list, rng, infantry_counts, kill_xp_delta,
                               state_views=None):
    """[M] flat event lists - runs every one of these M battles to
    completion (structure phase, archer phase, then up to
    MAX_ROUNDS_SAFETY_CAP rounds), in place. decide_target_list: length
    B, one {faction: (state_b, hex_index, faction) -> target_or_None}
    dict per batch item. infantry_counts/kill_xp_delta: [B, F] tensors
    the caller maintains and applies across the WHOLE turn's battle
    phase, not just these M battles (see turn.py's battle-phase
    orchestration for why - the dismount infantry cap and kill-XP awards
    are shared per-turn state, same as engine_old). state_views: optional
    pre-computed unstack_states(state) - see turn.py's run_turn docstring
    for why passing this in (once per turn, from the caller) beats every
    battle-slot iteration recomputing its own copy."""
    apply_structure_defense_shots(state, batch_idx, hex_idx, rng, infantry_counts, kill_xp_delta)
    apply_archer_abilities(state, batch_idx, hex_idx, rng, infantry_counts, kill_xp_delta)

    if state_views is None:
        state_views = unstack_states(state)  # cheap views (torch slices), stay live across mutations
    active = torch.ones(len(batch_idx), dtype=torch.bool, device=state.device)
    for _ in range(MAX_ROUNDS_SAFETY_CAP):
        active = active & ~is_battle_over_batch(state, batch_idx, hex_idx)
        if not bool(active.any()):
            break
        idx = torch.nonzero(active, as_tuple=False).flatten()
        b, h = batch_idx[idx], hex_idx[idx]
        alive_mask = _alive_counts(state, b, h) > 0
        target_choice = _gather_target_choices(state, b, h, decide_target_list, state_views, alive_mask)
        resolve_round_batch(state, b, h, target_choice, rng, infantry_counts, kill_xp_delta)


def rectify_overflow_batch(state, batch_idx, hex_idx, winner_faction, send_back_list, cap):
    """[M] flat event lists - winner_faction: [M] long. send_back_list:
    length M, one list of {"origin_hex", "units"} dicts per event
    (already obtained from each game's decide_rectification callback by
    the caller - see turn.py). cap: [M] long tensor or a single int (0
    forces full eviction - a foreign capital, see turn.py's battle-phase
    orchestration). After a battle resolves, if the winning stack exceeds
    `cap`, the winner sends the excess back to their own contributing
    origin hexes; units whose origin is invalid/unaccounted-for/full are
    trimmed (infantry -> cavalry -> archers) rather than left above cap.
    City ownership is NOT touched here - see turn.py for why."""
    M = len(batch_idx)
    device = state.device
    if not torch.is_tensor(cap):
        cap = torch.full((M,), cap, dtype=torch.long, device=device)

    idx_m = torch.arange(M, device=device)
    winning_units = faction_totals_sparse(
        state.battle_faction[batch_idx, hex_idx], state.battle_units[batch_idx, hex_idx], state.num_factions
    )[idx_m, winner_faction].clone()  # [M, 3]

    max_entries = max((len(sb) for sb in send_back_list), default=0)
    for e in range(max_entries):
        origin = torch.full((M,), -1, dtype=torch.long, device=device)
        entry_units = torch.zeros(M, 3, dtype=winning_units.dtype, device=device)
        for m in range(M):
            if e < len(send_back_list[m]):
                entry = send_back_list[m][e]
                o = entry["origin_hex"]
                if o is not None and 0 <= o < state.num_hexes:
                    origin[m] = o
                entry_units[m] = torch.tensor(entry["units"], device=device, dtype=entry_units.dtype)

        take = torch.minimum(entry_units, winning_units)
        winning_units = winning_units - take
        valid_origin = origin >= 0
        has_take = (take.sum(dim=-1) > 0) & valid_origin
        if not bool(has_take.any()):
            continue

        b_all, h_all = batch_idx, origin.clamp(min=0)
        is_empty = state.army_faction[b_all, h_all] == NO_FACTION
        claim = has_take & is_empty
        if bool(claim.any()):
            cb, ch = b_all[claim], h_all[claim]
            state.army_faction[cb, ch] = winner_faction[claim].to(state.army_faction.dtype)
            state.army_units[cb, ch] = 0

        is_own_now = state.army_faction[b_all, h_all] == winner_faction
        deposit_mask = has_take & is_own_now
        if bool(deposit_mask.any()):
            db, dh = b_all[deposit_mask], h_all[deposit_mask]
            dtake = take[deposit_mask]
            for ut in range(3):
                room = (MAX_STACK_SIZE - state.army_units[db, dh].sum(dim=-1)).clamp(min=0)
                deposit = torch.minimum(dtake[:, ut], room)
                state.army_units[db, dh, ut] += deposit

    total_remaining = winning_units.sum(dim=-1)
    excess = (total_remaining - cap).clamp(min=0)
    for ut in DEATH_PRIORITY:
        take = torch.minimum(winning_units[:, ut], excess)
        winning_units[:, ut] -= take
        excess = excess - take

    has_survivors = winning_units.sum(dim=-1) > 0
    if bool(has_survivors.any()):
        si = torch.nonzero(has_survivors, as_tuple=False).flatten()
        state.army_faction[batch_idx[si], hex_idx[si]] = winner_faction[si].to(state.army_faction.dtype)
        state.army_units[batch_idx[si], hex_idx[si]] = winning_units[si]
    if bool((~has_survivors).any()):
        ei = torch.nonzero(~has_survivors, as_tuple=False).flatten()
        state.army_faction[batch_idx[ei], hex_idx[ei]] = NO_FACTION
        state.army_units[batch_idx[ei], hex_idx[ei]] = 0
    state.frozen[batch_idx, hex_idx] = False
    state.locked[batch_idx, hex_idx] = False

    state.battle_faction[batch_idx, hex_idx] = NO_FACTION
    state.battle_origin[batch_idx, hex_idx] = NO_ORIGIN
    state.battle_units[batch_idx, hex_idx] = 0
    state.battle_moved[batch_idx, hex_idx] = False
    state.battle_round[batch_idx, hex_idx] = 0
    for m in range(M):
        b, h = int(batch_idx[m]), int(hex_idx[m])
        state.battle_order[b].remove(h)

    return state


# ---------------------------------------------------------------------------
# Single-game convenience wrappers, for agent callbacks (decide_target,
# etc.) that read battle state the same way engine_old's agents did - see
# module docstring for why faction order here is ascending index rather
# than "first battle-slot appearance" (RNG parity with engine_old was
# never on the table, so there's no reason to keep that ordering
# convention). `state` must have batch_size 1 (a per-game view - see
# state.py's unstack_states, used throughout this module to get one).

def faction_totals(state, hex_index):
    """{faction: int[3] tensor} for every faction with a contribution
    slot at hex_index (including ones currently at 0 units - see
    faction_alive_totals for the alive-only version), ascending faction
    index order."""
    bf = state.battle_faction[0, hex_index]
    present = [f for f in range(state.num_factions) if bool((bf == f).any())]
    if not present:
        return {}
    totals = faction_totals_sparse(bf.unsqueeze(0), state.battle_units[0, hex_index].unsqueeze(0), state.num_factions)[0]
    return {f: totals[f] for f in present}


def faction_moved_totals(state, hex_index):
    """Like faction_totals, but summed only over contribution slots
    flagged battle_moved=True - see state.py's battle_moved docstring."""
    bf = state.battle_faction[0, hex_index]
    present = [f for f in range(state.num_factions) if bool((bf == f).any())]
    if not present:
        return {}
    totals = faction_moved_totals_sparse(
        bf.unsqueeze(0), state.battle_units[0, hex_index].unsqueeze(0), state.battle_moved[0, hex_index].unsqueeze(0),
        state.num_factions,
    )[0]
    return {f: totals[f] for f in present}


def faction_alive_totals(state, hex_index):
    """{faction: total_units} for factions still alive in this battle."""
    return {f: t for f, t in faction_totals(state, hex_index).items() if int(t.sum()) > 0}


def get_legal_target_actions(state, hex_index, faction):
    """Valid targets for `faction` this round: any other faction still
    alive in the battle."""
    totals = faction_totals(state, hex_index)
    return [f for f in totals if f != faction and int(totals[f].sum()) > 0]


def is_battle_over(state, hex_index):
    return len(faction_alive_totals(state, hex_index)) <= 1


def get_winner(state, hex_index):
    alive = faction_alive_totals(state, hex_index)
    if len(alive) == 1:
        return next(iter(alive))
    return None
