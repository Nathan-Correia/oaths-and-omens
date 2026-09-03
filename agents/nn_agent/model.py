"""
HexPolicyNet: the actual network for agents/nn_agent - a small, per-hex
MLP encoder plus a handful of lightweight decision heads. Deliberately
simple (see package docstring): this is a shape to iterate on once real
training exists, not a considered architecture.

ONE network, shared across every faction (self-play convention - the
eventual trained policy is meant to play as whichever faction it's
asked to, not have a separate copy per faction). This works because
every feature below is FACTION-RELATIVE (mine/enemy/neutral, never a raw
faction id), so the same weights produce a sensible-shaped answer no
matter which faction is asking.

Feature design: one small vector per hex (terrain, ownership relation,
capital/upgrade status, army presence+composition, locked/frozen) plus a
small per-faction scalar vector (gold/kill-XP/VP/resources, all
normalized to roughly unit scale) - see _hex_features/_faction_features.
No positional/adjacency structure is fed in at all (not even hex
coordinates) - this network cannot "see" the board's shape, only what's
sitting on each hex in isolation, which is a real, deliberate limitation
of keeping this first pass simple (a graph/conv-style encoder that mixes
neighbor information would be the natural next step, not attempted here).
"""

import torch
import torch.nn as nn

from engine.state import MAX_STACK_SIZE, NO_FACTION, TERRAIN_TYPES, UPGRADE_TYPES

HEX_FEATURE_DIM = len(TERRAIN_TYPES) + 3 + 1 + (len(UPGRADE_TYPES) + 1) + 2 + 3 + 1 + 1  # 5+3+1+4+2+3+1+1 = 20
FACTION_FEATURE_DIM = 7  # gold, kill_xp, victory_points, wood, iron, clay, fish


def hex_features(state, faction):
    """[B, N, HEX_FEATURE_DIM] float - one row per (game, hex), entirely
    faction-relative (see module docstring). Batch-native: `state` can be
    any batch size, including the batch_size=1 per-game views every
    agents/*.py callback normally receives (old callers indexed `[0]`
    into what used to be an unbatched [N, ...] return - they now index
    `[0]` into the leading batch dim of this batched return instead, see
    __init__.py). Kept batch-native (rather than two separate code paths)
    so the batched decide_*.batch fast path (see __init__.py's
    nn_buy_batch/nn_movement_batch) can run ONE forward pass across every
    game in the engine's batch instead of one per game."""
    device = state.device
    terrain = state.terrain.long()  # [B, N]
    B, N = terrain.shape
    terrain_onehot = torch.zeros(B, N, len(TERRAIN_TYPES), device=device)
    terrain_onehot.scatter_(-1, terrain.unsqueeze(-1), 1.0)

    owner = state.city_owner  # [B, N]
    mine = owner == faction
    enemy = (owner != NO_FACTION) & ~mine
    neutral = owner == NO_FACTION
    owner_onehot = torch.stack([mine, enemy, neutral], dim=-1).float()

    is_capital = state.is_capital.float().unsqueeze(-1)

    upgrade = state.outpost_upgrade  # [B, N]
    upgrade_onehot = torch.zeros(B, N, len(UPGRADE_TYPES) + 1, device=device)
    has_upgrade = upgrade >= 0
    idx = torch.where(has_upgrade, upgrade.long() + 1, torch.zeros_like(upgrade.long()))
    upgrade_onehot.scatter_(-1, idx.unsqueeze(-1), 1.0)
    # rows where has_upgrade is False got a spurious 1 at column 0 from the
    # scatter above AND belong there anyway (index 0 means "no upgrade") -
    # rows where has_upgrade is True also got column 0 zeroed correctly by
    # scatter overwriting only their own idx column, so no further fixup
    # needed; a hex with idx=0 (no upgrade) and idx=upgrade+1 (has one) are
    # mutually exclusive by construction of `idx` above.

    army_faction = state.army_faction  # [B, N]
    army_mine = (army_faction == faction).float().unsqueeze(-1)
    army_enemy = ((army_faction != NO_FACTION) & (army_faction != faction)).float().unsqueeze(-1)
    army_units_norm = state.army_units.float() / MAX_STACK_SIZE

    locked = state.locked.float().unsqueeze(-1)
    frozen = state.frozen.float().unsqueeze(-1)

    return torch.cat(
        [terrain_onehot, owner_onehot, is_capital, upgrade_onehot, army_mine, army_enemy, army_units_norm, locked, frozen],
        dim=-1,
    )


def faction_features(state, faction):
    """[B, FACTION_FEATURE_DIM] float - this faction's own economy per
    game, roughly normalized to unit scale (the normalizers are rough
    references, not hard caps - e.g. gold can exceed 50 - just enough to
    keep values in a sane range for an untrained network's random
    weights). Batch-native - see hex_features."""
    return torch.stack([
        state.gold[:, faction].float() / 50.0,
        state.kill_xp[:, faction].float() / 10.0,
        state.victory_points[:, faction].float() / 50.0,
        state.resources[:, faction, 0].float() / 10.0,
        state.resources[:, faction, 1].float() / 10.0,
        state.resources[:, faction, 2].float() / 10.0,
        state.resources[:, faction, 3].float() / 10.0,
    ], dim=-1)


class HexPolicyNet(nn.Module):
    def __init__(self, hidden_dim=64):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.hex_encoder = nn.Sequential(
            nn.Linear(HEX_FEATURE_DIM, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
        )
        self.faction_encoder = nn.Sequential(nn.Linear(FACTION_FEATURE_DIM, hidden_dim), nn.ReLU())

        # per-hex heads (applied to every hex's own encoded vector):
        self.direction_head = nn.Linear(hidden_dim, 6)  # movement/cavalry: which of 6 directions
        self.location_head = nn.Linear(hidden_dim, 1)   # generic "how good is this hex as MY target
                                                          # hex" score - reused for outpost placement,
                                                          # infantry/conversion siting, setup placement/draft

        # tiny head for battle targeting - scores one candidate rival at a
        # time from a 3-feature summary, not the full hex/context encoder
        # (a battle target is a FACTION, not a hex - nothing hex-shaped to
        # feed in here).
        self.rival_head = nn.Linear(3, 1)

        # heads over pooled context (mean hex encoding + faction encoding):
        ctx_dim = hidden_dim * 2
        self.outpost_type_head = nn.Linear(ctx_dim, 3)     # none / build / upgrade
        self.outpost_unit_head = nn.Linear(ctx_dim, 3)     # which unit build_outpost sacrifices
        self.upgrade_head = nn.Linear(ctx_dim, 3)          # barracks / workshop / temple
        self.convert_type_head = nn.Linear(ctx_dim, 2)     # cavalry / archers preference
        self.resource_head = nn.Linear(ctx_dim, 2)         # iron / fish
        self.swap_head = nn.Linear(ctx_dim, 2)             # keep / swap

    def encode(self, state, faction):
        """(hex_hidden [B,N,H], ctx [B,2H]) - the two shared intermediates
        every decision head reads from. Recomputed fresh per call (no
        cross-call caching - see agents/nn_agent/__init__.py's docstring)
        but batch-native: called once on a batch_size=1 per-game view (the
        per-game decide_* callbacks) it does exactly what the old
        unbatched version did, just with an extra leading size-1 dim the
        caller indexes away with `[0]`; called once on the FULL
        multi-game state (the decide_*.batch fast path) it runs one
        forward pass for every game at once instead of one per game."""
        feats = hex_features(state, faction)  # [B,N,20]
        hex_hidden = self.hex_encoder(feats)  # [B,N,H]
        hex_pool = hex_hidden.mean(dim=1)  # [B,H]
        faction_ctx = self.faction_encoder(faction_features(state, faction))  # [B,H]
        ctx = torch.cat([hex_pool, faction_ctx], dim=-1)  # [B,2H]
        return hex_hidden, ctx
