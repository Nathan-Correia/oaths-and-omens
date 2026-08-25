"""
The policy network: one shared trunk + seven heads (movement, cavalry, buy,
target, rectification, capital preference, value).

Trunk: a per-hex projection (each hex's features, plus the global features
broadcast to every hex, through one Linear+ReLU) followed by a few rounds
of message-passing over the board's hex-adjacency graph
(state.grid.neighbor_table) - each round lets a hex's embedding absorb a
summary of its neighbors', so after N rounds a hex "knows about" roughly
its radius-N neighborhood, not just itself. This replaces the old
pure-per-hex-MLP trunk (no cross-hex mixing at all), which structurally
couldn't compare "this spot vs. that spot" - a real limitation for
placement/draft (farthest-point reasoning) and tactical movement alike.
No attention/transformer, no torch_geometric dependency - just plain
tensor ops over the neighbor table the engine already computes, which is
the right level of complexity for a ~150-300 hex board with 6 neighbors
per hex.

Heads read off the trunk's output:
  - movement/cavalry/buy/rectify/capital_pref: one small Dense layer
    applied independently to every hex's embedding (shared weights across
    hexes) - same shape as the corresponding legal-action mask, so
    logits and mask always line up 1:1.
  - target: only meaningful for one specific hex (the battle being
    decided) - takes that hex's embedding plus the pooled whole-board
    summary.
  - value: takes the pooled whole-board summary (mean over all hex
    embeddings) - a single number, not tied to any hex. Unused this
    milestone (no training loop exists yet) - present so a future PPO
    value function doesn't need an architecture change to appear.

capital_pref is a new head, shared across the three setup-phase
decisions (see actions.py's capital_choice_* functions) - placement,
draft, and swap are all fundamentally "how much do I want this hex as my
capital," just over different candidate subsets. It's deliberately its
own head rather than reusing rectify_head: those answer different
questions (which of my own battle-contribution origins gets overflow,
vs. how much I want a hex as my capital), and once a training loop
exists, sharing would let gradient signal from one corrupt the other.

`battle_hex_index` is always required (not optional) so every call has
the same shape/trace signature - pass 0 when it's not relevant (the
target head's output just goes unused in that case).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class HexMessagePassing(nn.Module):
    """One round of GraphSAGE-mean-style message passing: every hex's new
    embedding is a function of its own current embedding plus the mean of
    its on-board neighbors' embeddings (off-board neighbor slots, marked
    -1 in neighbor_table, are excluded from that mean rather than treated
    as a real zero-embedding neighbor)."""

    def __init__(self, hidden_dim):
        super().__init__()
        self.combine = nn.Linear(hidden_dim * 2, hidden_dim)

    def forward(self, embeddings, neighbor_table):
        """embeddings: float[num_hexes, hidden_dim]. neighbor_table:
        int64[num_hexes, 6], -1 for off-board."""
        valid = neighbor_table >= 0  # [num_hexes, 6]
        neighbor_embed = embeddings[neighbor_table.clamp(min=0)] * valid.unsqueeze(-1)
        neighbor_count = valid.sum(dim=1, keepdim=True).clamp(min=1)  # every real board
        # hex has >=1 neighbor, so this floor is defensive insurance, not a real case
        neighbor_mean = neighbor_embed.sum(dim=1) / neighbor_count
        return F.relu(self.combine(torch.cat([embeddings, neighbor_mean], dim=-1)))


class PolicyNetwork(nn.Module):
    def __init__(self, num_factions, hidden_dim=128, num_mp_rounds=2):
        super().__init__()
        self.num_factions = num_factions
        self.hidden_dim = hidden_dim

        # Imported here (not at module level) to avoid a hard import-time
        # dependency loop: encode.py doesn't need network.py, but
        # PER_HEX_FEATURES/GLOBAL_FEATURES are the source of truth for
        # this layer's input width.
        from .encode import GLOBAL_FEATURES, PER_HEX_FEATURES

        self.input_proj = nn.Linear(PER_HEX_FEATURES + GLOBAL_FEATURES, hidden_dim)
        self.mp_rounds = nn.ModuleList([HexMessagePassing(hidden_dim) for _ in range(num_mp_rounds)])

        self.movement_head = nn.Linear(hidden_dim, 6)
        self.cavalry_head = nn.Linear(hidden_dim, 6)
        self.buy_head = nn.Linear(hidden_dim, 5)
        self.rectify_head = nn.Linear(hidden_dim, 1)
        self.capital_pref_head = nn.Linear(hidden_dim, 1)
        self.target_head = nn.Linear(hidden_dim * 2, num_factions + 1)
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(self, per_hex, global_feats, neighbor_table, battle_hex_index):
        """per_hex: float[num_hexes, encode.PER_HEX_FEATURES]
        global_feats: float[encode.GLOBAL_FEATURES]
        neighbor_table: int64[num_hexes, 6], -1 for off-board
        battle_hex_index: int, which hex's embedding the target head
          should look at (irrelevant/unused outside battle decisions).

        Returns a dict of logits (and one plain value), all raw/unmasked
        - see actions.py for turning these into legal moves.
        """
        num_hexes = per_hex.shape[0]
        broadcast_global = global_feats.unsqueeze(0).expand(num_hexes, -1)
        x = torch.cat([per_hex, broadcast_global], dim=-1)

        embeddings = F.relu(self.input_proj(x))
        for mp_round in self.mp_rounds:
            embeddings = mp_round(embeddings, neighbor_table)

        pooled = embeddings.mean(dim=0)

        target_input = torch.cat([embeddings[battle_hex_index], pooled], dim=-1)

        return {
            "movement": self.movement_head(embeddings),
            "cavalry": self.cavalry_head(embeddings),
            "buy": self.buy_head(embeddings),
            "rectify": self.rectify_head(embeddings).squeeze(-1),
            "capital_pref": self.capital_pref_head(embeddings).squeeze(-1),
            "target": self.target_head(target_input),
            "value": self.value_head(pooled).squeeze(-1),
        }


def build_network(num_factions, hidden_dim=128, num_mp_rounds=2, seed=None, device=None):
    """Constructs a randomly-initialized PolicyNetwork. Unlike Flax's
    explicit-key .init() (which needed a dummy forward pass to discover
    shapes), torch Linear layers have static dims - PER_HEX_FEATURES/
    GLOBAL_FEATURES/hidden_dim, independent of num_hexes - so no
    num_hexes argument or dummy input is needed here.

    `seed`, if given, is applied via torch.manual_seed() immediately
    before construction for reproducible initial weights - nn.Linear's
    reset_parameters() doesn't take a per-call generator the way Flax's
    .init() took an explicit key, so this is a narrow, deliberate touch
    of torch's global RNG scoped to just this one call. The actual
    per-decision RNG (see agent.py) uses its own torch.Generator and
    never touches global state."""
    if seed is not None:
        torch.manual_seed(seed)
    network = PolicyNetwork(num_factions=num_factions, hidden_dim=hidden_dim, num_mp_rounds=num_mp_rounds)
    if device is not None:
        network = network.to(device)
    return network
