"""
NNAgent for engine: a randomly-initialized, UNTRAINED PyTorch policy
(see model.py's HexPolicyNet) wired into the same 9-callback shape every
other agents/*.py module exposes - referenced but not yet built when
run.py's docstring first mentioned it ("a randomly-initialized, untrained
PyTorch policy network is referenced in agents/__init__.py's docstring as
agents/nn_agent/, but that module doesn't exist yet"). This is that
module, now filled in.

SCOPE: this is deliberately NOT a training setup - no loss, no
optimizer, no self-play data collection, nothing that updates the
network's weights. It exists to prove the wiring works end to end (a
real torch.nn.Module making real forward-pass decisions can be dropped
into run.py/tournament.py exactly like any hand-written heuristic agent)
and to give something to actually train later, once that infrastructure
exists. With random weights, expect it to play badly - the point right
now is "does it run and produce legal-ish decisions," not "is it good."

ARCHITECTURE: intentionally simple (see model.py's HexPolicyNet
docstring) - a per-hex MLP with no spatial/adjacency structure, small
decision heads per callback. Expected to change once real training
starts driving architecture choices; nothing here should be read as a
consid­ered design.

PER-GAME BY DEFAULT, WITH AN OPTIONAL BATCHED FAST PATH: every callback
here still has a per-game form (state_view, faction, ...) -> decision,
matching every other agents/*.py module and engine/turn.py's standard
calling convention (see its module docstring for why agents stay
per-game Python functions even though the engine itself is batched).
That's what runs when this agent is mixed with other agent types in the
same batch, or driven one game at a time (run.py/tournament.py).

But calling a tiny network once per (game, faction, decision) - one
encode() forward pass per call - turned out to dominate the cost of
using this agent at any real batch size (profiled: ~4.5s of an 11.7s
CPU profile was inside encode(), and unbatched per-decision GPU syncs
made large batches on CUDA pathologically slow). So decide_buy,
decide_movement, and decide_cavalry (the three called every turn that
profiled as the dominant cost - decide_target/decide_resource_choice
are comparatively rare/cheap and were left per-game) additionally
expose a `.batch` attribute: a (state, faction, ...) -> length-B list of
decisions function that runs ONE forward pass across every game in the
full multi-game state at once (see nn_buy_batch/nn_movement_batch, and
model.py's encode - it's batch-native, not two separate code paths).
engine/turn.py's _gather_buy_decisions/_gather_movement_actions pick
this up automatically whenever the same callable is used for every
batch item (always true here - see make_nn_agents) and fall back to the
per-game loop otherwise, so mixed-agent tournaments are unaffected.

ONE network shared by every faction (self-play convention) - see
model.py for why the feature encoding is faction-relative, which is
what makes sharing one set of weights across every seat sensible.

decide_rectification is NOT network-driven: reuses random_agent's
random_rectification as-is. Producing a variable-length list of
{"origin_hex", "units"} entries that sum to a specific overflow amount
is a genuinely awkward shape for a fixed-size network head, and
rectification is post-battle bookkeeping, not a strategic decision worth
spending this first pass's complexity budget on - a real head can
replace this once there's a reason to.
"""

import random

import torch

from .model import HexPolicyNet
from ..random_agent import random_rectification
from engine.battle import faction_totals, get_legal_target_actions
from engine.buy import eligible_outpost_mask
from engine.state import NO_FACTION


def _device_of(state):
    return state.device


def _ensure_device(net, state):
    """Moves `net`'s parameters to `state`'s device the first time they
    disagree - lets one network instance serve games on whatever device
    they happen to live on (CPU for run.py/tournament.py's single-game
    use today; would matter more if this were ever driven from GPU-batched
    states directly)."""
    device = _device_of(state)
    if next(net.parameters()).device != device:
        net.to(device)


def _encode_single(net, state, faction):
    """net.encode is batch-native (see model.py) - this indexes `[0]` to
    get back the unbatched (hex_hidden [N,H], ctx [2H]) shape every
    per-game decide_* callback below expects, for the batch_size=1 view
    (state_views[b]) those callbacks are always called with."""
    hex_hidden, ctx = net.encode(state, faction)
    return hex_hidden[0], ctx[0]


def nn_movement(net, state, faction, legal_mask):
    """Shared by decide_movement/decide_cavalry - the mask's shape is
    the only difference (legal_cavalry_mask vs legal_movement_mask),
    already handled by the caller."""
    _ensure_device(net, state)
    if not bool(legal_mask.any()):
        return None
    with torch.no_grad():
        hex_hidden, _ = _encode_single(net, state, faction)
        logits = net.direction_head(hex_hidden)  # [N, 6]
    masked = logits.masked_fill(~legal_mask, float("-inf"))
    best = int(masked.flatten().argmax())
    N_dirs = masked.shape[1]
    return best // N_dirs, best % N_dirs


def nn_target(net, state, hex_index, faction):
    """Vectorized across `legal` (was: one Python-level float()/rival_head
    call - each an individual GPU sync when on CUDA - per candidate
    rival). Now builds the whole [K,3] feature matrix and runs
    rival_head once, with a single final sync to extract the winner."""
    _ensure_device(net, state)
    legal = get_legal_target_actions(state, hex_index, faction)
    if not legal:
        return None
    totals = faction_totals(state, hex_index)
    device = state.device
    own_total = (totals[faction].sum().float() / 6.0) if faction in totals else torch.zeros((), device=device)
    with torch.no_grad():
        rival_totals = torch.stack([totals[f].float().sum() for f in legal]) / 6.0  # [K]
        feat = torch.stack(
            [rival_totals, own_total.expand(len(legal)), torch.ones(len(legal), device=device)], dim=-1,
        )  # [K, 3]
        scores = net.rival_head(feat).squeeze(-1)  # [K]
    best_i = int(scores.argmax())
    return legal[best_i]


def nn_resource_choice(net, state, faction):
    _ensure_device(net, state)
    with torch.no_grad():
        _, ctx = _encode_single(net, state, faction)
        choice = int(net.resource_head(ctx).argmax())
    return "iron" if choice == 0 else "fish"


def nn_placement(net, state, faction, legal_mask):
    _ensure_device(net, state)
    with torch.no_grad():
        hex_hidden, _ = _encode_single(net, state, faction)
        scores = net.location_head(hex_hidden).squeeze(-1)
    masked = scores.masked_fill(~legal_mask, float("-inf"))
    return int(masked.argmax())


def nn_draft(net, state, faction, legal_pool):
    """Vectorized (was: one float() sync per candidate in `legal_pool`)."""
    _ensure_device(net, state)
    with torch.no_grad():
        hex_hidden, _ = _encode_single(net, state, faction)
        scores = net.location_head(hex_hidden).squeeze(-1)
    idx = torch.tensor(legal_pool, dtype=torch.long, device=state.device)
    best_local = int(scores[idx].argmax())
    return legal_pool[best_local]


def nn_swap(net, state, faction, leftover_hex, placer_hex):
    """swap_head's pooled-context vote is combined with a cheap, nearly-
    free signal the pooled context alone can't see: whether the network's
    own per-hex location_head - the same one that drives where it places/
    drafts in the first place - already rates placer_hex (what forcing
    the swap would win) above leftover_hex (what declining leaves it
    with)."""
    _ensure_device(net, state)
    with torch.no_grad():
        hex_hidden, ctx = _encode_single(net, state, faction)
        location_scores = net.location_head(hex_hidden).squeeze(-1)
        head_vote = int(net.swap_head(ctx).argmax()) == 1
    prefers_placer_hex = bool(location_scores[placer_hex] > location_scores[leftover_hex])
    return head_vote or prefers_placer_hex


def nn_buy(net, state, faction):
    """Builds the fixed-shape buy decision (see engine/buy.py's module
    docstring): WHICH hex, via the shared location_head, gated to
    whatever's actually legal there (an outpost action needs an eligible
    hex with a spare unit; infantry needs an owned, unlocked city; a
    conversion needs an owned hex with infantry) - the network never sees
    illegal options to begin with, same "propose from a legal set"
    shape every hand-written agent uses. Quantities (how much infantry,
    how many conversions) are NOT network decisions here - they're
    derived the same way greedy_buy's are (spend what's affordable /
    available), keeping this first pass's scope to "which hex, which
    type," not "how much" - see module docstring."""
    _ensure_device(net, state)
    with torch.no_grad():
        hex_hidden, ctx = _encode_single(net, state, faction)
        location_scores = net.location_head(hex_hidden).squeeze(-1)
        outpost_type = int(net.outpost_type_head(ctx).argmax())
        outpost_unit = int(net.outpost_unit_head(ctx).argmax())
        upgrade = int(net.upgrade_head(ctx).argmax())
        want_cavalry = int(net.convert_type_head(ctx).argmax()) == 0

    decision = {}
    gold = int(state.gold[0, faction])

    # Each candidate set below used to be a Python list filtered/argmax'd
    # via `max(candidates, key=lambda h: float(location_scores[h]))` - one
    # GPU sync PER CANDIDATE HEX (up to N of them). Replaced with a single
    # boolean mask + masked_fill + argmax per decision - one sync total,
    # regardless of how many hexes qualify.
    if outpost_type == 1 and gold >= 3:
        eligible = eligible_outpost_mask(state, faction)[0]
        army_mask = (state.army_faction[0] == faction) & ~state.locked[0] & eligible
        if bool(army_mask.any()):
            best_h = int(location_scores.masked_fill(~army_mask, float("-inf")).argmax())
            unit_counts = state.army_units[0, best_h]  # [3]
            available_mask = unit_counts > 0
            if bool(available_mask.any()):
                unit_type = outpost_unit if bool(available_mask[outpost_unit]) else int(available_mask.long().argmax())
                decision["outpost_type"] = 1
                decision["outpost_hex"] = best_h
                decision["outpost_unit_type"] = unit_type
    elif outpost_type == 2:
        outpost_mask = (
            (state.city_owner[0] == faction) & ~state.is_capital[0] & ~state.locked[0] & (state.outpost_upgrade[0] < 0)
        )
        if bool(outpost_mask.any()):
            best_h = int(location_scores.masked_fill(~outpost_mask, float("-inf")).argmax())
            decision["outpost_type"] = 2
            decision["outpost_hex"] = best_h
            decision["outpost_upgrade"] = upgrade

    city_mask = (state.city_owner[0] == faction) & ~state.locked[0]
    if gold >= 2 and bool(city_mask.any()):
        best_h = int(location_scores.masked_fill(~city_mask, float("-inf")).argmax())
        decision["infantry_buy"] = {best_h: gold // 2}

    army_mask = (state.army_faction[0] == faction) & (state.army_units[0, :, 0] > 0)
    kill_xp = int(state.kill_xp[0, faction])
    if kill_xp > 0 and bool(army_mask.any()):
        best_h = int(location_scores.masked_fill(~army_mask, float("-inf")).argmax())
        if want_cavalry:
            decision["convert_cavalry"] = {best_h: kill_xp}
        else:
            decision["convert_archers"] = {best_h: kill_xp}

    return decision


def nn_movement_batch(net, state, faction, legal_mask):
    """Batched counterpart to nn_movement: ONE forward pass across every
    game in `state` (not a per-game view - the full multi-game batch)
    instead of B separate encode() calls. legal_mask: [B,N,6], already
    computed once for the whole batch by turn.py's
    _gather_movement_actions (see its docstring) - passed straight
    through to the masked argmax below. Returns a length-B list of
    (hex_index, direction) or None, matching what B calls to nn_movement
    would have returned.

    Wired up as decide_movement.batch/decide_cavalry.batch in
    make_nn_agents - see engine/turn.py's _gather_movement_actions for
    where this fast path gets picked up (only when the SAME callable is
    used for every batch item, which is always true here since one
    HexPolicyNet is shared across the whole batch and every faction)."""
    _ensure_device(net, state)
    B = state.batch_size
    with torch.no_grad():
        hex_hidden, _ = net.encode(state, faction)  # [B,N,H]
        logits = net.direction_head(hex_hidden)  # [B,N,6]
    N_dirs = logits.shape[-1]
    masked = logits.masked_fill(~legal_mask, float("-inf"))
    any_legal = legal_mask.reshape(B, -1).any(dim=-1)
    best = masked.reshape(B, -1).argmax(dim=-1)  # [B] - one sync for the whole batch
    best_hex = (best // N_dirs).tolist()
    best_dir = (best % N_dirs).tolist()
    any_legal_list = any_legal.tolist()
    return [(best_hex[b], best_dir[b]) if any_legal_list[b] else None for b in range(B)]


def nn_buy_batch(net, state, faction):
    """Batched counterpart to nn_buy: same decision logic (see its
    docstring), but as ONE forward pass + a handful of whole-batch masked
    argmaxes across every game in `state`, instead of B separate
    encode() calls each doing its own small argmaxes. Returns a length-B
    list of decision dicts (see nn_buy's return shape; an empty dict
    means "no action" for that game, same as nn_buy returning {})."""
    _ensure_device(net, state)
    device = state.device
    B = state.batch_size
    with torch.no_grad():
        hex_hidden, ctx = net.encode(state, faction)  # [B,N,H], [B,2H]
        location_scores = net.location_head(hex_hidden).squeeze(-1)  # [B,N]
        outpost_type = net.outpost_type_head(ctx).argmax(dim=-1)  # [B]
        outpost_unit = net.outpost_unit_head(ctx).argmax(dim=-1)  # [B]
        upgrade = net.upgrade_head(ctx).argmax(dim=-1)  # [B]
        want_cavalry = net.convert_type_head(ctx).argmax(dim=-1) == 0  # [B] bool

    gold = state.gold[:, faction]  # [B]
    kill_xp = state.kill_xp[:, faction]  # [B]
    arange_b = torch.arange(B, device=device)
    decisions = [{} for _ in range(B)]

    build_want = (outpost_type == 1) & (gold >= 3)
    if bool(build_want.any()):
        eligible = eligible_outpost_mask(state, faction)  # [B,N]
        army_mask = (state.army_faction == faction) & ~state.locked & eligible  # [B,N]
        can_build = build_want & army_mask.any(dim=-1)
        if bool(can_build.any()):
            best_h = location_scores.masked_fill(~army_mask, float("-inf")).argmax(dim=-1)  # [B]
            available_mask = state.army_units[arange_b, best_h] > 0  # [B,3]
            has_unit = available_mask.any(dim=-1)
            fallback_unit = available_mask.long().argmax(dim=-1)  # first-available, matches nn_buy's `available[0]`
            wants_available = available_mask[arange_b, outpost_unit]
            chosen_unit = torch.where(wants_available, outpost_unit, fallback_unit)
            do_build = can_build & has_unit
            for b in torch.nonzero(do_build, as_tuple=False).flatten().tolist():
                decisions[b]["outpost_type"] = 1
                decisions[b]["outpost_hex"] = int(best_h[b])
                decisions[b]["outpost_unit_type"] = int(chosen_unit[b])

    upgrade_want = outpost_type == 2
    if bool(upgrade_want.any()):
        outpost_mask = (
            (state.city_owner == faction) & ~state.is_capital & ~state.locked & (state.outpost_upgrade < 0)
        )  # [B,N]
        can_upgrade = upgrade_want & outpost_mask.any(dim=-1)
        if bool(can_upgrade.any()):
            best_h = location_scores.masked_fill(~outpost_mask, float("-inf")).argmax(dim=-1)
            for b in torch.nonzero(can_upgrade, as_tuple=False).flatten().tolist():
                decisions[b]["outpost_type"] = 2
                decisions[b]["outpost_hex"] = int(best_h[b])
                decisions[b]["outpost_upgrade"] = int(upgrade[b])

    city_mask = (state.city_owner == faction) & ~state.locked  # [B,N]
    can_infantry = (gold >= 2) & city_mask.any(dim=-1)
    if bool(can_infantry.any()):
        best_h = location_scores.masked_fill(~city_mask, float("-inf")).argmax(dim=-1)
        for b in torch.nonzero(can_infantry, as_tuple=False).flatten().tolist():
            decisions[b]["infantry_buy"] = {int(best_h[b]): int(gold[b]) // 2}

    convert_army_mask = (state.army_faction == faction) & (state.army_units[..., 0] > 0)  # [B,N]
    can_convert = (kill_xp > 0) & convert_army_mask.any(dim=-1)
    if bool(can_convert.any()):
        best_h = location_scores.masked_fill(~convert_army_mask, float("-inf")).argmax(dim=-1)
        for b in torch.nonzero(can_convert, as_tuple=False).flatten().tolist():
            key = "convert_cavalry" if bool(want_cavalry[b]) else "convert_archers"
            decisions[b][key] = {int(best_h[b]): int(kill_xp[b])}

    return decisions


def make_nn_agents(num_factions, seed=0):
    """Same 9-callback shape as every other make_X_agents. One
    HexPolicyNet, randomly initialized from `seed` (torch.manual_seed
    before construction - reproducible per seed, same convention every
    other agent's `seed`-keyed rngs follow) and shared across every
    faction (self-play convention - see module docstring)."""
    torch.manual_seed(seed)
    net = HexPolicyNet()
    net.eval()
    rngs = {f: random.Random(seed * 1_000_003 + f) for f in range(num_factions)}

    def decide_buy(state, faction):
        return nn_buy(net, state, faction)

    decide_buy.batch = lambda state, faction: nn_buy_batch(net, state, faction)

    def decide_movement(state, faction, step, legal_mask):
        return nn_movement(net, state, faction, legal_mask)

    decide_movement.batch = lambda state, faction, step, legal_mask: nn_movement_batch(net, state, faction, legal_mask)

    def decide_cavalry(state, faction, step, legal_mask):
        return nn_movement(net, state, faction, legal_mask)

    decide_cavalry.batch = lambda state, faction, step, legal_mask: nn_movement_batch(net, state, faction, legal_mask)

    def decide_target(state, hex_index, faction):
        return nn_target(net, state, hex_index, faction)

    def decide_rectification(state, hex_index, winner_faction, cap):
        return random_rectification(state, hex_index, winner_faction, cap, rngs[winner_faction])

    def decide_resource_choice(state, faction, hex_index):
        return nn_resource_choice(net, state, faction)

    def decide_placement(state, faction, legal_mask):
        return nn_placement(net, state, faction, legal_mask)

    def decide_draft(state, faction, legal_pool):
        return nn_draft(net, state, faction, legal_pool)

    def decide_swap(state, faction, leftover_hex, placer_faction, placer_hex):
        return nn_swap(net, state, faction, leftover_hex, placer_hex)

    factions = range(num_factions)
    return (
        {f: decide_buy for f in factions},
        {f: decide_movement for f in factions},
        {f: decide_cavalry for f in factions},
        {f: decide_target for f in factions},
        {f: decide_rectification for f in factions},
        {f: decide_resource_choice for f in factions},
        {f: decide_placement for f in factions},
        {f: decide_draft for f in factions},
        {f: decide_swap for f in factions},
    )
