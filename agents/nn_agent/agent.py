"""
Wires the policy network into engine.turn.run_turn's callback
interface, self-play style: ONE shared (network, params) plays every
faction, each call re-encoding the board from that faction's own point
of view (see encode.py) - not five separately-behaving closures, just
the same functions called with a different `faction` argument each time.

JAX's RNG is purely functional (no global "next random number" the way
Python's `random` module has) - callers have to explicitly split a key
every time they want fresh randomness. _KeySource is a small,
deliberately-impure wrapper around that: it holds one "current" key and
hands out a fresh split-off key on every call, so the rest of this file
(and engine.turn's callback interface, which knows nothing about
JAX) can just ask for "the next key" instead of threading key state
through every function signature by hand. The actual JAX computations
underneath stay pure given whatever key they're handed - only this
bookkeeping layer is stateful, which is a standard pattern for
non-jitted driver code.

The network's forward pass IS jitted here (jax.jit(network.apply)),
unlike everything else in this file: `network` is closed over (not
passed as a traced argument), so this compiles once per distinct input
shape (i.e. once per board radius actually used) and every subsequent
call - regardless of which faction, decision type, or battle hex - reuses
that compiled version. encode_observation and actions.py's masking/
decoding stay plain numpy/Python: encode_observation builds its arrays
via boolean masking with a data-dependent size, which isn't jit-shaped
the way a fixed dense-layer stack is, and neither of them is the
bottleneck anyway (see the conversation this was added from - profiling
showed essentially all per-decision cost was the un-jitted forward
pass, not encoding/decoding).
"""

import jax

from engine.battle import get_legal_target_actions
from engine.buy import get_legal_buy_actions  # noqa: F401 - documents the expected legal-action source

from . import actions
from .encode import encode_observation


class _KeySource:
    def __init__(self, seed):
        self._key = jax.random.PRNGKey(seed)

    def next(self):
        self._key, sub_key = jax.random.split(self._key)
        return sub_key


def make_nn_agents(network, params, num_factions, seed=0, max_turns=100):
    """Returns (decide_buy, decide_movement, decide_cavalry,
    decide_target, decide_rectification) dicts - each {faction:
    callable}, matching engine.turn.run_turn's expected signatures,
    all backed by the same (network, params)."""
    keys = _KeySource(seed)
    apply_fn = jax.jit(network.apply)

    def forward(state, faction, battle_hex_index=0):
        per_hex, global_feats = encode_observation(state, faction, max_turns=max_turns)
        return apply_fn(params, per_hex, global_feats, battle_hex_index)

    def decide_buy(state, faction, legal):
        out = forward(state, faction)
        mask = actions.buy_action_mask(state, legal)
        return actions.decode_buy(keys.next(), out["buy"], mask)

    def decide_movement(state, faction, step, legal_mask):
        out = forward(state, faction)
        return actions.decode_movement_or_cavalry(keys.next(), out["movement"], legal_mask)

    def decide_cavalry(state, faction, step, legal_mask):
        out = forward(state, faction)
        return actions.decode_movement_or_cavalry(keys.next(), out["cavalry"], legal_mask)

    def decide_target(state, hex_index, faction):
        out = forward(state, faction, battle_hex_index=hex_index)
        legal = get_legal_target_actions(state, hex_index, faction)
        mask = actions.target_mask(num_factions, legal)
        return actions.decode_target(keys.next(), out["target"], mask)

    def decide_rectification(state, hex_index, winner_faction, cap):
        out = forward(state, winner_faction, battle_hex_index=hex_index)
        return actions.decode_rectification(keys.next(), out["rectify"], state, hex_index, winner_faction, cap)

    factions = range(num_factions)
    return (
        {f: decide_buy for f in factions},
        {f: decide_movement for f in factions},
        {f: decide_cavalry for f in factions},
        {f: decide_target for f in factions},
        {f: decide_rectification for f in factions},
    )
