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
"""

import numpy as np
import torch

from engine.battle import get_legal_target_actions

from . import actions
from .encode import encode_observation


def make_nn_agents(network, num_factions, seed=0, max_turns=100, device=None):
    """Returns (decide_buy, decide_movement, decide_cavalry,
    decide_target, decide_rectification, decide_placement, decide_draft,
    decide_swap) dicts - each {faction: callable}, matching
    engine.turn.run_turn's and engine.placement.run_city_setup's
    expected signatures, all backed by the same shared `network`."""
    device = device or torch.device("cpu")
    network = network.to(device).eval()
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)

    # HexGrid never changes within a game (see engine/geometry.py) - the
    # int64 neighbor-table tensor is cheap to build but not free, and
    # every decision this turn/game needs the same one, so cache it per
    # grid instance rather than reconverting every call.
    neighbor_table_cache = {}

    def _neighbor_table(state):
        key = id(state.grid)
        cached = neighbor_table_cache.get(key)
        if cached is None:
            cached = torch.from_numpy(state.grid.neighbor_table.astype(np.int64)).to(device)
            neighbor_table_cache[key] = cached
        return cached

    @torch.inference_mode()
    def forward(state, faction, battle_hex_index=0):
        per_hex, global_feats = encode_observation(state, faction, max_turns=max_turns)
        per_hex_t = torch.from_numpy(per_hex).to(device)
        global_feats_t = torch.from_numpy(global_feats).to(device)
        return network(per_hex_t, global_feats_t, _neighbor_table(state), battle_hex_index)

    def decide_buy(state, faction, legal):
        out = forward(state, faction)
        mask = actions.buy_action_mask(state, legal)
        return actions.decode_buy(generator, out["buy"], mask, legal)

    def decide_movement(state, faction, step, legal_mask):
        out = forward(state, faction)
        return actions.decode_movement_or_cavalry(generator, out["movement"], legal_mask)

    def decide_cavalry(state, faction, step, legal_mask):
        out = forward(state, faction)
        return actions.decode_movement_or_cavalry(generator, out["cavalry"], legal_mask)

    def decide_target(state, hex_index, faction):
        out = forward(state, faction, battle_hex_index=hex_index)
        legal = get_legal_target_actions(state, hex_index, faction)
        mask = actions.target_mask(num_factions, legal)
        return actions.decode_target(generator, out["target"], mask)

    def decide_rectification(state, hex_index, winner_faction, cap):
        out = forward(state, winner_faction, battle_hex_index=hex_index)
        return actions.decode_rectification(generator, out["rectify"], state, hex_index, winner_faction, cap)

    def decide_placement(state, faction, legal_mask):
        out = forward(state, faction)
        return actions.decode_capital_choice(generator, out["capital_pref"], legal_mask)

    def decide_draft(state, faction, legal_pool):
        out = forward(state, faction)
        mask = actions.capital_choice_mask_from_pool(state.num_hexes, legal_pool)
        return actions.decode_capital_choice(generator, out["capital_pref"], mask)

    def decide_swap(state, faction, leftover_hex, placer_faction, placer_hex):
        out = forward(state, faction)
        mask = actions.capital_choice_mask_pair(state.num_hexes, leftover_hex, placer_hex)
        chosen = actions.decode_capital_choice(generator, out["capital_pref"], mask)
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
