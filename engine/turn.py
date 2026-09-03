"""
Batched turn orchestration for engine.

Agent callback shape: every decide_* argument is now a length-B list, one
{faction: callable} dict per batch item (matching the convention used
throughout movement.py/battle.py/collect.py/buy.py - agents stay per-game
Python functions; see the plan's "Consequence for agents/tooling"
section). Per-callback signatures, in the new fixed-shape/batched terms:

  decide_buy(state_b, faction) -> {"outpost_type", "outpost_hex",
    "outpost_unit_type", "outpost_upgrade", "infantry_buy", "convert_cavalry",
    "convert_archers"} - see buy.py's module docstring for the fixed-shape
    action fields.
  decide_movement(state_b, faction, step) -> (hex_index, direction) or None
  decide_cavalry(state_b, faction, step) -> (hex_index, direction) or None
  decide_target(state_b, hex_index, faction) -> target_faction or None
  decide_rectification(state_b, hex_index, winner_faction, cap) ->
    [{"origin_hex", "units"}, ...]
  decide_resource_choice(state_b, faction, hex_index) -> "iron" or "fish"

RULE (unchanged from engine_old): Buy -> Movement -> Combat -> Collect.

Battle-phase orchestration is the one piece of this module that isn't a
straightforward "wire the batched phases together" - see
_run_battle_phase's docstring for the battle-slot-index design (loop over
"each game's Nth pending battle", not over games, so every game's own
battle_order sequencing - see state.py's docstring for why that order is
real game state, not cosmetic - is preserved while still batching across
games at each slot).
"""

import torch

from .battle import faction_totals, get_winner_batch, rectify_overflow_batch, resolve_full_battle_batch
from .buy import _count_all_units_in_play_batch, apply_buy_phase_batch
from .collect import OUTPOST_DESTROY_VP, VP_TO_WIN, apply_collect_phase
from .movement import actions_from_dicts, apply_movement_step, legal_cavalry_mask, legal_movement_mask
from .state import MAX_STACK_SIZE, NO_FACTION, NO_UPGRADE, RESOURCE_TYPES, UPGRADE_TYPES, unstack_states
from .terrain import apply_terrain_effects

MOVEMENT_STEPS = 3
CAVALRY_STEPS = 2

CHECKPOINT_LABELS = ["Start", "Buy", "Move 1", "Move 2", "Move 3", "Cav 1", "Cav 2", "Battle"]


def _uniform_batched_callable(decide_list, f, B):
    """If decide_list[b][f] is the SAME callable object for every b (the
    normal case whenever one agent instance is reused across the whole
    batch - e.g. nn_agent's shared HexPolicyNet, or any agent used
    uniformly in a self-play batch) AND that callable exposes a `.batch`
    attribute (see agents/nn_agent's module docstring), return it so the
    caller can run ONE call across the whole batch instead of B calls.
    Returns None otherwise (mixed-agent batches, or agents with no
    batched form) - callers fall back to the per-game loop, unchanged."""
    fn0 = decide_list[0][f]
    batch_fn = getattr(fn0, "batch", None)
    if batch_fn is None:
        return None
    if not all(decide_list[b][f] is fn0 for b in range(1, B)):
        return None
    return batch_fn


def _gather_buy_decisions(state, decide_buy_list, state_views=None):
    """Calls each game's per-faction decide_buy callback and stacks the
    results into the fixed-shape batched tensors apply_buy_phase_batch
    wants. Missing/partial fields in a callback's returned dict default
    to "no action" (all-zero counts, outpost_type 0) - matches the
    "invalid/absent answer silently falls back to doing nothing" pattern
    used throughout engine (movement/battle callbacks returning None,
    placement/draft falling back to a random legal choice).

    Per faction, uses the batched fast path (see _uniform_batched_callable)
    when available instead of B separate per-game calls - see
    agents/nn_agent's module docstring for why this exists.

    state_views: optional pre-computed unstack_states(state) - pass this
    in (see run_turn) rather than letting this recompute it, since the
    views are just torch slices that stay live across mutations; the
    original per-call unstack_states was itself a smaller instance of
    the same "recompute a cheap-to-share thing on every call" waste the
    legal-mask fix addressed - see run_turn's docstring."""
    B, N = state.terrain.shape
    F = state.num_factions
    device = state.device
    if state_views is None:
        state_views = unstack_states(state)

    outpost_type = torch.zeros(B, F, dtype=torch.long, device=device)
    outpost_hex = torch.full((B, F), -1, dtype=torch.long, device=device)
    outpost_unit_type = torch.zeros(B, F, dtype=torch.long, device=device)
    outpost_upgrade = torch.zeros(B, F, dtype=torch.long, device=device)
    infantry_buy = torch.zeros(B, F, N, dtype=torch.long, device=device)
    convert_cavalry = torch.zeros(B, F, N, dtype=torch.long, device=device)
    convert_archers = torch.zeros(B, F, N, dtype=torch.long, device=device)

    for f in range(F):
        batch_fn = _uniform_batched_callable(decide_buy_list, f, B)
        if batch_fn is not None:
            decisions = batch_fn(state, f)
        else:
            decisions = [decide_buy_list[b][f](state_views[b], f) for b in range(B)]
        for b, decision in enumerate(decisions):
            if not decision:
                continue
            if decision.get("outpost_type"):
                outpost_type[b, f] = int(decision["outpost_type"])
                outpost_hex[b, f] = int(decision.get("outpost_hex", -1))
                outpost_unit_type[b, f] = int(decision.get("outpost_unit_type", 0))
                outpost_upgrade[b, f] = int(decision.get("outpost_upgrade", 0))
            for hex_index, count in decision.get("infantry_buy", {}).items():
                infantry_buy[b, f, hex_index] += int(count)
            for hex_index, count in decision.get("convert_cavalry", {}).items():
                convert_cavalry[b, f, hex_index] += int(count)
            for hex_index, count in decision.get("convert_archers", {}).items():
                convert_archers[b, f, hex_index] += int(count)

    return outpost_type, outpost_hex, outpost_unit_type, outpost_upgrade, infantry_buy, convert_cavalry, convert_archers


def _gather_movement_actions(state, decide_list, step, legal_mask_fn, state_views=None):
    """Calls each game's per-faction movement/cavalry callback for this
    step and returns the [{faction: action_or_None}, ...] shape
    movement.actions_from_dicts expects. legal_mask_fn: legal_movement_mask
    or legal_cavalry_mask - called ONCE PER FACTION over the whole batch
    (each call is already a proper batched [B, N, 6] op), not once per
    (batch item, faction) pair - calling it per batch item was the
    dominant cost of a whole turn (profiled: ~95% of run_turn's wall
    time at B=128, from re-deriving the same per-faction legality mask
    from scratch B times over instead of slicing one batched computation
    - the exact "engine primitive computed correctly but called
    wastefully" shape as this session's earlier _can_build_outpost
    finding, just in the batched engine instead of the original one).

    state_views: optional pre-computed unstack_states(state) - see
    _gather_buy_decisions' docstring for why passing this in beats
    recomputing it (this function used to call it fresh every one of the
    5 times per turn it's used - 3 movement steps + 2 cavalry steps)."""
    B = state.batch_size
    F = state.num_factions
    if state_views is None:
        state_views = unstack_states(state)
    full_masks = [legal_mask_fn(state, f) for f in range(F)]  # each [B, N, 6], one batched call per faction

    per_faction_actions = []
    for f in range(F):
        batch_fn = _uniform_batched_callable(decide_list, f, B)
        if batch_fn is not None:
            per_faction_actions.append(batch_fn(state, f, step, full_masks[f]))
        else:
            per_faction_actions.append(
                [decide_list[b][f](state_views[b], f, step, full_masks[f][b]) for b in range(B)]
            )

    return [{f: per_faction_actions[f][b] for f in range(F)} for b in range(B)]


def _run_battle_phase(state, decide_target_list, decide_rectification_list, rng, state_views=None):
    """Resolves every pending battle across the whole batch, in place.
    Battle-slot-index design (see module docstring): loop over slot =
    0, 1, 2, ... - "each game's (slot)-th still-pending battle, in that
    game's own battle_order" - resolving that whole cross-game slice
    together via battle.py's batched primitives before moving to the
    next slot. This reproduces each individual game's own sequential
    battle-resolution order (the thing that actually matters - see
    state.py's battle_order docstring for the shared infantry-dismount-
    cap coupling across battles in the SAME game) while still batching
    across games at every slot; a game with only 1 pending battle simply
    has nothing to do once its own single slot is resolved."""
    B = state.batch_size
    F = state.num_factions
    device = state.device
    if state_views is None:
        state_views = unstack_states(state)

    counts_all = _count_all_units_in_play_batch(state)  # [B, F, 3]
    infantry_counts = counts_all[..., 0].clone()  # running tally shared across every battle THIS turn
    kill_xp_delta = torch.zeros(B, F, dtype=torch.long, device=device)

    pending = [list(state.battle_order[b]) for b in range(B)]  # snapshot, like engine_old's pending_hexes
    max_slots = max((len(p) for p in pending), default=0)

    for slot in range(max_slots):
        batch_list, hex_list = [], []
        for b in range(B):
            if slot < len(pending[b]):
                h = pending[b][slot]
                if bool(state.locked[b, h]):
                    batch_list.append(b)
                    hex_list.append(h)
        if not batch_list:
            continue
        batch_idx = torch.tensor(batch_list, dtype=torch.long, device=device)
        hex_idx = torch.tensor(hex_list, dtype=torch.long, device=device)

        resolve_full_battle_batch(
            state, batch_idx, hex_idx, decide_target_list, rng, infantry_counts, kill_xp_delta, state_views=state_views,
        )
        winner = get_winner_batch(state, batch_idx, hex_idx)  # [M] long, NO_FACTION if no winner

        has_winner = winner != NO_FACTION
        if bool(has_winner.any()):
            wi = torch.nonzero(has_winner, as_tuple=False).flatten()
            wb, wh, ww = batch_idx[wi], hex_idx[wi], winner[wi]
            owner = state.city_owner[wb, wh]
            owner_present = owner != NO_FACTION
            owner_differs = owner_present & (owner != ww)
            is_cap_hex = state.is_capital[wb, wh]

            cap = torch.full((len(wi),), MAX_STACK_SIZE, dtype=torch.long, device=device)
            cap[owner_differs & is_cap_hex] = 0

            destroy = owner_differs & ~is_cap_hex
            if bool(destroy.any()):
                db, dh, dw = wb[destroy], wh[destroy], ww[destroy]
                state.city_owner[db, dh] = NO_FACTION
                state.victory_points.index_put_(
                    (db, dw), torch.full((len(db),), OUTPOST_DESTROY_VP, dtype=state.victory_points.dtype, device=device),
                    accumulate=True,
                )

            send_back_list = []
            for i in range(len(wi)):
                b, h, w, c = int(wb[i]), int(wh[i]), int(ww[i]), int(cap[i])
                send_back_list.append(decide_rectification_list[b][w](state_views[b], h, w, c))
            rectify_overflow_batch(state, wb, wh, ww, send_back_list, cap)

        no_winner = ~has_winner
        if bool(no_winner.any()):
            ni = torch.nonzero(no_winner, as_tuple=False).flatten()
            nb_, nh = batch_idx[ni], hex_idx[ni]
            state.army_faction[nb_, nh] = NO_FACTION
            state.army_units[nb_, nh] = 0
            state.locked[nb_, nh] = False
            state.battle_faction[nb_, nh] = NO_FACTION
            state.battle_units[nb_, nh] = 0
            state.battle_moved[nb_, nh] = False
            state.battle_round[nb_, nh] = 0
            for i in ni.tolist():
                b, h = int(batch_idx[i]), int(hex_idx[i])
                state.battle_order[b].remove(h)

    state.kill_xp += kill_xp_delta
    return state


def get_game_winner(state):
    """[B] long - NO_FACTION until some faction's VP total has reached
    VP_TO_WIN in that game. Among every faction at or above VP_TO_WIN,
    the strict highest total wins outright; an exact tie for the top
    total is broken by whoever placed their capital LATER (see
    placement.py's run_city_setup - a single incrementing counter, so no
    two factions can ever tie on it)."""
    top = state.victory_points.max(dim=-1).values  # [B]
    contenders = (state.victory_points == top.unsqueeze(-1)) & (top.unsqueeze(-1) >= VP_TO_WIN)  # [B, F]
    settle_order_masked = torch.where(contenders, state.capital_settle_order, torch.full_like(state.capital_settle_order, -1))
    winner = settle_order_masked.argmax(dim=-1)
    has_winner = (top >= VP_TO_WIN) & contenders.any(dim=-1)
    return torch.where(has_winner, winner.to(state.victory_points.dtype), torch.full_like(top, NO_FACTION))


def check_game_end(state, max_turns=None):
    """[B] bool - True once a game should stop: the VP win condition has
    been hit (see get_game_winner), or (if max_turns is given - purely an
    infra safety net, not part of the rules) that game's turn_number has
    reached it."""
    ended = get_game_winner(state) != NO_FACTION
    if max_turns is not None:
        ended = ended | (state.turn_number >= max_turns)
    return ended


def run_turn(state, decide_buy_list, decide_movement_list, decide_cavalry_list, decide_target_list,
             decide_rectification_list, decide_resource_choice_list, rng):
    """Mutates and returns `state`. Each decide_*_list argument is length
    B, one {faction: callable} dict per batch item - see module docstring
    for each callback's signature. `rng`: a torch.Generator on the same
    device as `state`.

    Every batch item advances one full turn together, regardless of
    whether some have already won (see check_game_end/get_game_winner) -
    a finished game just keeps producing decisions/mutations nobody reads
    (VP_TO_WIN is a floor, not a ceiling, and re-running a phase on an
    already-won game is harmless, just wasted work). Skipping ended
    batch items - and any auto-reset of them - is the future training
    loop's concern (see the plan's "explicitly deferred" section), not
    this function's.

    state_views (unstack_states(state)) is computed ONCE here and
    threaded through every phase that needs a per-game view for agent
    callbacks, rather than each phase recomputing it - the views are
    just torch slices sharing storage with `state`, so they correctly
    reflect every phase's mutations without needing to be refreshed (see
    state.py's unstack_states docstring); recomputing it 6+ times a turn
    was itself a smaller instance of the same "recompute a cheap-to-
    share thing on every call" waste the legal-mask fix addressed."""
    F = state.num_factions
    state_views = unstack_states(state)

    buy_args = _gather_buy_decisions(state, decide_buy_list, state_views)
    apply_buy_phase_batch(state, *buy_args)

    for step in range(MOVEMENT_STEPS):
        actions = _gather_movement_actions(state, decide_movement_list, step, legal_movement_mask, state_views)
        from_hex, direction, has_action = actions_from_dicts(actions, F, state.device)
        apply_movement_step(state, from_hex, direction, has_action, rng, cavalry_only=False)

    for step in range(CAVALRY_STEPS):
        actions = _gather_movement_actions(state, decide_cavalry_list, step, legal_cavalry_mask, state_views)
        from_hex, direction, has_action = actions_from_dicts(actions, F, state.device)
        apply_movement_step(state, from_hex, direction, has_action, rng, cavalry_only=True)

    _run_battle_phase(state, decide_target_list, decide_rectification_list, rng, state_views)
    apply_terrain_effects(state)
    apply_collect_phase(state, decide_resource_choice_list)

    state.turn_number += 1
    return state


def _snapshot_entry(state, hex_index, coord):
    """Per-hex log entry: {"q","r","s","city","troops","battle"}. "city"
    is None, or {"faction", "is_capital", "upgrade"}. "upgrade" is one of
    UPGRADE_TYPES or None. Single-game (batch item 0) only - see
    snapshot_hexes."""
    if int(state.city_owner[0, hex_index]) != NO_FACTION:
        upgrade_index = int(state.outpost_upgrade[0, hex_index])
        city = {
            "faction": int(state.city_owner[0, hex_index]),
            "is_capital": bool(state.is_capital[0, hex_index]),
            "upgrade": UPGRADE_TYPES[upgrade_index] if upgrade_index != NO_UPGRADE else None,
        }
    else:
        city = None
    entry = {
        "q": coord[0], "r": coord[1], "s": coord[2],
        "city": city,
        "troops": None, "battle": None,
    }
    if bool(state.locked[0, hex_index]):
        totals = faction_totals(state, hex_index)
        if totals:
            entry["battle"] = {
                "contributions": [
                    {"faction": f, "infantry": int(t[0]), "cavalry": int(t[1]), "archers": int(t[2])}
                    for f, t in totals.items()
                ]
            }
    elif int(state.army_faction[0, hex_index]) != NO_FACTION:
        entry["troops"] = {
            "faction": int(state.army_faction[0, hex_index]),
            "infantry": int(state.army_units[0, hex_index, 0]),
            "cavalry": int(state.army_units[0, hex_index, 1]),
            "archers": int(state.army_units[0, hex_index, 2]),
            "frozen": bool(state.frozen[0, hex_index]),
        }
    return entry


def snapshot_hexes(state):
    """Full per-hex snapshot (all coords), single-game (batch item 0)."""
    return [_snapshot_entry(state, i, coord) for i, coord in enumerate(state.grid.coords)]


def sparse_hexes(full_hexes):
    """Filters a snapshot down to only hexes with something on them."""
    return [e for e in full_hexes if e["city"] is not None or e["troops"] is not None or e["battle"] is not None]


def _player_stats_snapshot(state):
    return {
        f: {
            "gold": int(state.gold[0, f]),
            "resources": {r: int(state.resources[0, f, i]) for i, r in enumerate(RESOURCE_TYPES)},
            "kill_xp": int(state.kill_xp[0, f]),
            "victory_points": int(state.victory_points[0, f]),
            "alive": bool(state.alive[0, f]),
        }
        for f in range(state.num_factions)
    }


def run_turn_and_log(state, decide_buy_list, decide_movement_list, decide_cavalry_list, decide_target_list,
                      decide_rectification_list, decide_resource_choice_list, rng):
    """Same as run_turn, but also returns a turn_record capturing enough
    to replay/visualize the turn: a full board snapshot at every one of
    the turn's checkpoints (see CHECKPOINT_LABELS), and player stats at
    each checkpoint. Single-game (batch_size=1) only - run.py's use case.

    battle_events is NOT populated with per-roll detail here, unlike the
    pre-batching-rewrite engine: its only consumer was hex_visualizer.py's
    battle animation (compute_battle_table), which is deprecated -
    rebuilding that level of structured logging for the batched battle
    resolution (see battle.py's resolve_full_battle_batch, which tracks
    aggregate kill-XP but not a round-by-round event log) wasn't worth
    doing for a consumer that no longer exists. Left as an empty list
    (the field still exists, for callers that only check its presence)
    rather than removed, so board_state.json's shape doesn't change."""
    assert state.batch_size == 1, "run_turn_and_log is single-game only"
    F = state.num_factions
    turn_number = int(state.turn_number[0])

    # "Start" checkpoint: before this turn's Buy phase.
    checkpoints = [sparse_hexes(snapshot_hexes(state))]
    player_stats = [_player_stats_snapshot(state)]

    buy_args = _gather_buy_decisions(state, decide_buy_list)
    apply_buy_phase_batch(state, *buy_args)
    checkpoints.append(sparse_hexes(snapshot_hexes(state)))
    player_stats.append(_player_stats_snapshot(state))

    for step in range(MOVEMENT_STEPS):
        actions = _gather_movement_actions(state, decide_movement_list, step, legal_movement_mask)
        from_hex, direction, has_action = actions_from_dicts(actions, F, state.device)
        apply_movement_step(state, from_hex, direction, has_action, rng, cavalry_only=False)
        checkpoints.append(sparse_hexes(snapshot_hexes(state)))
        player_stats.append(_player_stats_snapshot(state))

    for step in range(CAVALRY_STEPS):
        actions = _gather_movement_actions(state, decide_cavalry_list, step, legal_cavalry_mask)
        from_hex, direction, has_action = actions_from_dicts(actions, F, state.device)
        apply_movement_step(state, from_hex, direction, has_action, rng, cavalry_only=True)
        checkpoints.append(sparse_hexes(snapshot_hexes(state)))
        player_stats.append(_player_stats_snapshot(state))

    _run_battle_phase(state, decide_target_list, decide_rectification_list, rng)
    apply_terrain_effects(state)
    apply_collect_phase(state, decide_resource_choice_list)
    checkpoints.append(sparse_hexes(snapshot_hexes(state)))
    player_stats.append(_player_stats_snapshot(state))

    state.turn_number += 1

    turn_record = {
        "turn_number": turn_number,
        "checkpoints": checkpoints,
        "battle_events": [],
        "player_stats": player_stats,
    }
    return state, turn_record
