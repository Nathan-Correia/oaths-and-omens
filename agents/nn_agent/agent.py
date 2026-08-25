"""
Wires the policy network into engine.turn.run_turn's and
engine.placement.run_city_setup's callback interfaces, self-play style:
ONE shared network plays every faction, each call re-encoding the board
from that faction's own point of view (see encode.py) - not eight
separately-behaving closures, just the same functions called with a
different `faction` argument each time. Covers all eight decision
points (the five turn-phase ones plus placement/draft/swap) - unlike
this package's previous JAX version, no driver-script-level fallback to
another agent's setup-phase policies is needed anymore.

RNG: one torch.Generator per made-agent-set, seeded once, threaded
explicitly through every sampling call (see actions.py) rather than
touching torch's global RNG state - matches this codebase's convention
everywhere else (every scripted agent seeds its own random.Random
per-faction). This is genuinely simpler than the JAX version's
_KeySource: a torch.Generator is already stateful, so every
torch.multinomial(..., generator=...) call both consumes and advances it
in place - no explicit "split off the next key" bookkeeping needed.

forward() is wrapped in torch.inference_mode() - this is pure inference
with no gradients needed at this milestone (no training loop exists
yet). It's called in eager mode, not torch.compile()'d: this is
unbatched, one-decision-at-a-time inference on a small network, where
torch.compile's per-shape compile overhead (and, on native Windows, a
less mature Triton backend) would be a bad trade. Worth revisiting once
a batched training loop actually makes throughput matter.

`recorder` (optional): when given a training.buffer.RolloutBuffer, every
decide_* closure pushes a step record into it as a side effect (per_hex/
global_feats actually used, the mask, the sampled action, its log-prob,
and the value-head estimate) while still returning exactly what the
engine's callback contract expects - so the exact same closures serve
both plain inference (recorder=None, e.g. run.py) and training rollout
collection. Deliberately uses plain string literals for each decision's
`decision_kind` ("movement"/"buy"/... - see each closure below) rather
than importing constants from training.buffer: agents/ has no import
dependency on training/ anywhere else, and shouldn't gain one just for
a handful of string constants - training/buffer.py defines the matching
constants for ITS OWN callers (gae.py/ppo.py) to use instead of
retyping string literals there.
"""

import numpy as np
import torch

from engine.battle import get_legal_target_actions

from . import actions
from .encode import encode_observation


def make_nn_agents(network, num_factions, seed=0, max_turns=100, device=None, recorder=None):
    """Returns (decide_buy, decide_movement, decide_cavalry,
    decide_target, decide_rectification, decide_placement, decide_draft,
    decide_swap) dicts - each {faction: callable}, matching
    engine.turn.run_turn's and engine.placement.run_city_setup's
    expected signatures, all backed by the same shared `network`.

    `recorder`: optional training.buffer.RolloutBuffer - see module
    docstring."""
    device = device or torch.device("cpu")
    network = network.to(device).eval()
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)

    # HexGrid(radius) is a pure function of radius (see engine/geometry.py)
    # - keying by radius rather than id(state.grid) means this stays
    # correct AND bounded (one entry per distinct radius ever seen) across
    # a long-running process that creates many fresh HexGrid instances
    # (e.g. one per rollout game), unlike id()-keying, which would grow
    # forever and risks a stale/wrong tensor if a garbage-collected grid's
    # id() gets reused by an unrelated object.
    neighbor_table_cache = {}

    def _neighbor_table(state):
        key = state.grid.radius
        cached = neighbor_table_cache.get(key)
        if cached is None:
            cached = torch.from_numpy(state.grid.neighbor_table.astype(np.int64)).to(device)
            neighbor_table_cache[key] = cached
        return cached

    @torch.inference_mode()
    def forward(state, faction, battle_hex_index=0):
        """network(...) always expects a batch dimension (see network.py's
        module docstring) - this wraps the single decision being made
        right now as a batch of 1, then immediately unwraps the result,
        so every decide_* closure below and all of actions.py can keep
        working with the same unbatched shapes as before."""
        per_hex, global_feats = encode_observation(state, faction, max_turns=max_turns)
        per_hex_t = torch.from_numpy(per_hex).to(device).unsqueeze(0)
        global_feats_t = torch.from_numpy(global_feats).to(device).unsqueeze(0)
        battle_hex_t = torch.tensor([battle_hex_index], dtype=torch.long, device=device)
        out = network(per_hex_t, global_feats_t, _neighbor_table(state), battle_hex_t)
        out = {k: v.squeeze(0) for k, v in out.items()}
        return out, per_hex, global_feats

    def decide_buy(state, faction, legal):
        out, per_hex, global_feats = forward(state, faction)
        mask = actions.buy_action_mask(state, legal)
        parsed, log_prob = actions.decode_buy(generator, out["buy"], mask, legal)
        if recorder is not None:
            recorder.record_step(
                faction=faction, decision_kind="buy", per_hex=per_hex, global_feats=global_feats,
                battle_hex_index=0, mask=mask,
                action_repr=actions.buy_action_to_choice_vector(state.num_hexes, parsed),
                log_prob=float(log_prob.item()), value=float(out["value"].item()),
            )
        return parsed

    def decide_movement(state, faction, step, legal_mask):
        out, per_hex, global_feats = forward(state, faction)
        parsed, log_prob = actions.decode_movement_or_cavalry(generator, out["movement"], legal_mask)
        if recorder is not None and log_prob is not None:
            hex_index, direction = parsed
            recorder.record_step(
                faction=faction, decision_kind="movement", per_hex=per_hex, global_feats=global_feats,
                battle_hex_index=0, mask=legal_mask, action_repr=hex_index * 6 + direction,
                log_prob=float(log_prob.item()), value=float(out["value"].item()),
            )
        return parsed

    def decide_cavalry(state, faction, step, legal_mask):
        out, per_hex, global_feats = forward(state, faction)
        parsed, log_prob = actions.decode_movement_or_cavalry(generator, out["cavalry"], legal_mask)
        if recorder is not None and log_prob is not None:
            hex_index, direction = parsed
            recorder.record_step(
                faction=faction, decision_kind="cavalry", per_hex=per_hex, global_feats=global_feats,
                battle_hex_index=0, mask=legal_mask, action_repr=hex_index * 6 + direction,
                log_prob=float(log_prob.item()), value=float(out["value"].item()),
            )
        return parsed

    def decide_target(state, hex_index, faction):
        out, per_hex, global_feats = forward(state, faction, battle_hex_index=hex_index)
        legal = get_legal_target_actions(state, hex_index, faction)
        mask = actions.target_mask(num_factions, legal)
        action, log_prob = actions.decode_target(generator, out["target"], mask)
        if recorder is not None:
            recorder.record_step(
                faction=faction, decision_kind="target", per_hex=per_hex, global_feats=global_feats,
                battle_hex_index=hex_index, mask=mask,
                action_repr=num_factions if action is None else action,
                log_prob=float(log_prob.item()), value=float(out["value"].item()),
            )
        return action

    def decide_rectification(state, hex_index, winner_faction, cap):
        out, per_hex, global_feats = forward(state, winner_faction, battle_hex_index=hex_index)
        send_back, log_prob = actions.decode_rectification(generator, out["rectify"], state, hex_index,
                                                             winner_faction, cap)
        if recorder is not None and log_prob is not None:
            mask = actions.rectification_origin_mask(state, hex_index, winner_faction)
            recorder.record_step(
                faction=winner_faction, decision_kind="rectify", per_hex=per_hex, global_feats=global_feats,
                battle_hex_index=hex_index, mask=mask, action_repr=send_back[0]["origin_hex"],
                log_prob=float(log_prob.item()), value=float(out["value"].item()),
            )
        return send_back

    def decide_placement(state, faction, legal_mask):
        out, per_hex, global_feats = forward(state, faction)
        chosen, log_prob = actions.decode_capital_choice(generator, out["capital_pref"], legal_mask)
        if recorder is not None:
            recorder.record_step(
                faction=faction, decision_kind="placement", per_hex=per_hex, global_feats=global_feats,
                battle_hex_index=0, mask=legal_mask, action_repr=chosen,
                log_prob=float(log_prob.item()), value=float(out["value"].item()),
            )
        return chosen

    def decide_draft(state, faction, legal_pool):
        out, per_hex, global_feats = forward(state, faction)
        mask = actions.capital_choice_mask_from_pool(state.num_hexes, legal_pool)
        chosen, log_prob = actions.decode_capital_choice(generator, out["capital_pref"], mask)
        if recorder is not None:
            recorder.record_step(
                faction=faction, decision_kind="draft", per_hex=per_hex, global_feats=global_feats,
                battle_hex_index=0, mask=mask, action_repr=chosen,
                log_prob=float(log_prob.item()), value=float(out["value"].item()),
            )
        return chosen

    def decide_swap(state, faction, leftover_hex, placer_faction, placer_hex):
        out, per_hex, global_feats = forward(state, faction)
        mask = actions.capital_choice_mask_pair(state.num_hexes, leftover_hex, placer_hex)
        chosen, log_prob = actions.decode_capital_choice(generator, out["capital_pref"], mask)
        if recorder is not None:
            recorder.record_step(
                faction=faction, decision_kind="swap", per_hex=per_hex, global_feats=global_feats,
                battle_hex_index=0, mask=mask, action_repr=chosen,
                log_prob=float(log_prob.item()), value=float(out["value"].item()),
            )
        return chosen == placer_hex

    factions = range(num_factions)
    return (
        {f: decide_buy for f in factions},
        {f: decide_movement for f in factions},
        {f: decide_cavalry for f in factions},
        {f: decide_target for f in factions},
        {f: decide_rectification for f in factions},
        {f: decide_placement for f in factions},
        {f: decide_draft for f in factions},
        {f: decide_swap for f in factions},
    )
