"""
The policy network: one shared trunk + six heads (movement, cavalry, buy,
target, rectification, value).

Trunk: a per-hex MLP - each hex's features (plus the global features,
broadcast to every hex) go through the same two-layer network
independently, producing one embedding vector per hex. There's no
cross-hex mixing (no attention, no message-passing along
neighbor_table) - a hex's embedding only knows about itself and the
game-wide globals, not what's happening at neighboring hexes. That's a
real ceiling on how sophisticated the resulting policy can get (it can't
directly reason "there's a threat two hexes away"), but it's a
deliberate simplification for getting the plumbing (masking, decoding,
self-play wiring) correct first - training quality is a separate,
later concern. A graph/attention layer over neighbor_table is the
natural upgrade when that starts to matter.

Heads read off the trunk's output:
  - movement/cavalry/buy/rectify: one small Dense layer applied
    independently to every hex's embedding (shared weights across
    hexes) - same shape as the corresponding legal-action mask, so
    logits and mask always line up 1:1.
  - target: only meaningful for one specific hex (the battle being
    decided) - takes that hex's embedding plus the pooled whole-board
    summary.
  - value: takes the pooled whole-board summary (mean over all hex
    embeddings) - a single number, not tied to any hex.

`battle_hex_index` is always required (not optional) so every call has
the same shape/trace signature - pass 0 when it's not relevant (the
target head's output just goes unused in that case). Not jitted yet;
that's part of the later batching work, not this milestone.
"""

import flax.linen as nn
import jax.numpy as jnp


class PolicyNetwork(nn.Module):
    num_factions: int
    hidden_dim: int = 128

    @nn.compact
    def __call__(self, per_hex, global_feats, battle_hex_index):
        """per_hex: float[num_hexes, encode.PER_HEX_FEATURES]
        global_feats: float[encode.GLOBAL_FEATURES]
        battle_hex_index: int, which hex's embedding the target head
          should look at (irrelevant/unused outside battle decisions).

        Returns a dict of logits (and one plain value), all raw/
        unmasked - see actions.py for turning these into legal moves.
        """
        num_hexes = per_hex.shape[0]
        broadcast_global = jnp.broadcast_to(global_feats, (num_hexes, global_feats.shape[0]))
        x = jnp.concatenate([per_hex, broadcast_global], axis=-1)

        x = nn.relu(nn.Dense(self.hidden_dim, name="trunk_1")(x))
        per_hex_embedding = nn.relu(nn.Dense(self.hidden_dim, name="trunk_2")(x))
        pooled = jnp.mean(per_hex_embedding, axis=0)

        movement_logits = nn.Dense(6, name="movement_head")(per_hex_embedding)
        cavalry_logits = nn.Dense(6, name="cavalry_head")(per_hex_embedding)
        buy_logits = nn.Dense(4, name="buy_head")(per_hex_embedding)
        rectify_logits = jnp.squeeze(nn.Dense(1, name="rectify_head")(per_hex_embedding), axis=-1)
        value = jnp.squeeze(nn.Dense(1, name="value_head")(pooled), axis=-1)

        target_input = jnp.concatenate([per_hex_embedding[battle_hex_index], pooled], axis=-1)
        target_logits = nn.Dense(self.num_factions + 1, name="target_head")(target_input)

        return {
            "movement": movement_logits,
            "cavalry": cavalry_logits,
            "buy": buy_logits,
            "rectify": rectify_logits,
            "target": target_logits,
            "value": value,
        }


def init_params(rng_key, num_hexes, num_factions, hidden_dim=128):
    """Random-initialized params for a given board size/faction count -
    the "random weights" starting point. Uses dummy zero inputs of the
    right shape purely to trace the network's structure; the actual
    values don't matter for initialization."""
    from . import encode

    network = PolicyNetwork(num_factions=num_factions, hidden_dim=hidden_dim)
    dummy_per_hex = jnp.zeros((num_hexes, encode.PER_HEX_FEATURES))
    dummy_global = jnp.zeros((encode.GLOBAL_FEATURES,))
    params = network.init(rng_key, dummy_per_hex, dummy_global, 0)
    return network, params
