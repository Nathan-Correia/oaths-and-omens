"""The policy network (PLAN.md §10.4-§10.6).

This mirrors engine/src/neural.cpp exactly, including the byte-record expansion,
because the weights trained here have to run there. The wire layout is defined in
engine/include/oo/encode.hpp; the constants below are copied from it and from
engine/include/oo/config.hpp.

IF YOU CHANGE ANYTHING HERE, change neural.cpp to match and bump
kEncodingVersion. A mismatch does not raise - it silently trains on misaligned
features, which looks exactly like a model that will not learn.

The intended end state is the reverse of this: define the model here only, export
TorchScript, and have C++ load it, so there is one definition rather than two.
That is a small refactor of neural.cpp and is not done yet.
"""

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

# --- constants mirrored from the engine -------------------------------------
MAX_HEXES = 331
MAX_FACTIONS = 10
NUM_UNIT_TYPES = 3
NUM_RESOURCES = 4
NUM_DIRECTIONS = 6
NUM_TERRAIN_TYPES = 5
MAX_STACK_SIZE = 6
MAX_BATTLE_CONTRIB = 16
VP_TO_WIN = 50
OUTPOST_CAP = 6

HEADER_BYTES = 16
FACTION_BYTES = 24
HEX_BYTES = 12
HEADER_OFF = 0
FACTION_OFF = HEADER_BYTES
HEX_OFF = FACTION_OFF + FACTION_BYTES * MAX_FACTIONS

HEX_FEATURES = 54
FACTION_FEATURES = 15
HEADER_FEATURES = 15

# Relative-distance bias buckets (§10.5).
DIST_BUCKETS = 9
BUCKET_HEX_FACTION = 9
BUCKET_HEX_CLS = 10
BUCKET_FACTION_FACTION = 11
BUCKET_FACTION_CLS = 12
BUCKET_CLS_CLS = 13
NUM_BUCKETS = 14

NUM_DECISION_KINDS = 9
BUY_VARIANTS = 9


def distance_bucket(d: int) -> int:
    """Log bucketing: 1 vs 2 matters a lot, 15 vs 16 barely at all."""
    if d <= 5:
        return d
    if d <= 7:
        return 6
    if d <= 11:
        return 7
    return 8


def hexes_for_radius(radius: int) -> int:
    return 3 * radius * (radius + 1) + 1


def read_i16(b: torch.Tensor, lo: int) -> torch.Tensor:
    """Little-endian int16 out of two uint8 lanes, sign-extended."""
    v = b[..., lo].to(torch.int32) + b[..., lo + 1].to(torch.int32) * 256
    return torch.where(v > 32767, v - 65536, v)



class Block(nn.Module):
    def __init__(self, d: int, heads: int, ffn_mult: int):
        super().__init__()
        self.n_heads = heads
        self.head_dim = d // heads
        self.norm1 = nn.LayerNorm(d)
        self.norm2 = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)
        self.fc1 = nn.Linear(d, d * ffn_mult)
        self.fc2 = nn.Linear(d * ffn_mult, d)

    def forward(self, x: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
        b, t, _ = x.shape
        h = self.norm1(x)
        qkv = self.qkv(h).view(b, t, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        # A FLOAT attn_mask is added to the scores, which is what a relative
        # position bias is. A bool mask would mean allowed/blocked instead.
        att = F.scaled_dot_product_attention(q, k, v, attn_mask=bias.unsqueeze(0))
        att = att.transpose(1, 2).reshape(b, t, self.n_heads * self.head_dim)
        x = x + self.proj(att)
        x = x + self.fc2(F.gelu(self.fc1(self.norm2(x))))
        return x


class Policy(nn.Module):
    """One instance serves one (radius, num_factions) bucket, as in C++."""

    def __init__(self, radius=7, num_factions=8, d_model=128, n_blocks=4, n_heads=4,
                 ffn_mult=4, stem_layers=2, neighbours=None, buckets=None):
        super().__init__()
        self.radius = radius
        self.num_factions = num_factions
        self.hexes = hexes_for_radius(radius)
        self.tokens = self.hexes + num_factions + 1
        self.d = d_model
        self.n_heads = n_heads

        # TorchScript cannot close over module-level globals, so every constant
        # the scripted methods touch has to live on the instance. Keeping them
        # here rather than inlining also means the layout is stated once.
        self.hex_off = HEX_OFF
        self.hex_bytes = HEX_BYTES
        self.faction_off = FACTION_OFF
        self.faction_bytes = FACTION_BYTES
        self.header_off = HEADER_OFF
        self.header_bytes = HEADER_BYTES
        self.max_factions = MAX_FACTIONS
        self.n_terrain = NUM_TERRAIN_TYPES
        self.stack_cap = float(MAX_STACK_SIZE)
        self.vp_to_win = float(VP_TO_WIN)
        self.outpost_cap = float(OUTPOST_CAP)
        self.n_resources = NUM_RESOURCES
        self.n_kinds = NUM_DECISION_KINDS
        self.n_unit_types = NUM_UNIT_TYPES

        self.hex_in = nn.Linear(HEX_FEATURES, d_model)
        self.faction_in = nn.Linear(FACTION_FEATURES, d_model)
        self.header_in = nn.Linear(HEADER_FEATURES, d_model)
        self.slot_emb = nn.Embedding(MAX_FACTIONS, d_model)
        self.stem = nn.ModuleList(
            [nn.Linear(d_model * 7, d_model) for _ in range(stem_layers)])
        self.blocks = nn.ModuleList(
            [Block(d_model, n_heads, ffn_mult) for _ in range(n_blocks)])
        self.bias_table = nn.Parameter(torch.zeros(n_heads, NUM_BUCKETS))

        self.head_move = nn.Linear(d_model, NUM_DIRECTIONS)
        self.head_pass = nn.Linear(d_model, 1)
        self.head_hex = nn.Linear(d_model, 1)
        self.head_buy = nn.Linear(d_model, BUY_VARIANTS)
        self.head_target = nn.Linear(d_model, 1)
        self.head_resource = nn.Linear(d_model, 2)
        self.head_swap = nn.Linear(d_model, 2)
        self.head_rect = nn.Linear(d_model, MAX_BATTLE_CONTRIB * NUM_UNIT_TYPES)
        self.head_value = nn.Linear(d_model, 1)

        # Geometry tables. Passed in by the caller (which reads them from the
        # engine's own grid) or built here from cube coordinates.
        if neighbours is None or buckets is None:
            neighbours, buckets = build_geometry(radius, num_factions)
        self.register_buffer("neighbours", neighbours, persistent=True)
        self.register_buffer("buckets", buckets, persistent=True)

    # --- byte record -> features ------------------------------------------
    # Kept TorchScript-compatible: no module-level constants, no f-strings, and
    # lists built by explicit append rather than comprehension. The feature
    # widths are checked in tests/check_parity.py instead of by an assert here,
    # since a scripted assert cannot carry a useful message.
    def embed(self, raw: torch.Tensor) -> torch.Tensor:
        b = raw.shape[0]
        i32 = raw.to(torch.int32)

        hx = i32[:, self.hex_off:self.hex_off + self.hex_bytes * self.hexes]
        hx = hx.view(b, self.hexes, self.hex_bytes)
        terrain = hx[:, :, 0]
        city = hx[:, :, 1]
        city_flags = hx[:, :, 2]
        army = hx[:, :, 3]
        units = hx[:, :, 4:7].float()
        flags = hx[:, :, 7]
        battle_round = hx[:, :, 8].float()
        battle_factions = hx[:, :, 10] + hx[:, :, 11] * 256

        hf = [F.one_hot(terrain.long(), self.n_terrain).float()]
        for k in range(7):
            hf.append(((flags >> k) & 1).float().unsqueeze(-1))
        hf.append(F.one_hot(city.long(), self.max_factions + 1).float())
        hf.append((city_flags & 1).float().unsqueeze(-1))
        hf.append(F.one_hot(((city_flags >> 1) & 3).long(), 4).float())
        hf.append(F.one_hot(army.long(), self.max_factions + 1).float())
        hf.append(units / self.stack_cap)
        hf.append((units.sum(-1) / self.stack_cap).unsqueeze(-1))
        hf.append((battle_round / 5.0).unsqueeze(-1))
        for k in range(self.max_factions):
            hf.append(((battle_factions >> k) & 1).float().unsqueeze(-1))
        # The expansion above always produces float32 (one_hot, the /
        # divisions), but the weights may have been converted to half for
        # inference. Cast at each projection rather than trying to build the
        # features in the weight dtype - integer ops have no half equivalent.
        wdt = self.hex_in.weight.dtype
        hex_tok = self.hex_in(torch.cat(hf, -1).to(wdt))

        for layer in self.stem:
            zeros = torch.zeros(b, 1, self.d, device=hex_tok.device, dtype=hex_tok.dtype)
            padded = torch.cat([hex_tok, zeros], 1)
            gathered = padded.index_select(1, self.neighbours.view(-1))
            hex_tok = hex_tok + F.gelu(layer(gathered.view(b, self.hexes, 7 * self.d)))

        fc = i32[:, self.faction_off:self.faction_off + self.faction_bytes * self.max_factions]
        fc = fc.view(b, self.max_factions, self.faction_bytes)[:, :self.num_factions]
        ff = [fc[:, :, 0].float().unsqueeze(-1)]
        ff.append(fc[:, :, 1].float().unsqueeze(-1))
        ff.append((fc[:, :, 2].float() / self.outpost_cap).unsqueeze(-1))
        ff.append(fc[:, :, 3:6].float() / 20.0)
        ff.append((fc[:, :, 6].float() / 10.0).unsqueeze(-1))
        ff.append((read_i16(fc, 8).float() / 50.0).unsqueeze(-1))
        res = []
        for r in range(self.n_resources):
            res.append(read_i16(fc, 10 + 2 * r).float().unsqueeze(-1))
        ff.append(torch.cat(res, -1) / 20.0)
        ff.append((read_i16(fc, 18).float() / self.vp_to_win).unsqueeze(-1))
        ff.append((read_i16(fc, 20).float() / 20.0).unsqueeze(-1))
        ff.append((read_i16(fc, 22).float() / self.vp_to_win).unsqueeze(-1))
        slots = torch.arange(self.num_factions, device=raw.device)
        fac_tok = self.faction_in(torch.cat(ff, -1).to(wdt)) + self.slot_emb(slots).unsqueeze(0)

        hd = i32[:, self.header_off:self.header_off + self.header_bytes]
        gf = [(hd[:, 1].float() / 8.0).unsqueeze(-1)]
        gf.append((hd[:, 2].float() / 10.0).unsqueeze(-1))
        gf.append(F.one_hot(hd[:, 3].long(), self.n_kinds).float())
        gf.append((hd[:, 4].float() / 3.0).unsqueeze(-1))
        gf.append((read_i16(hd, 6).float() / 331.0).unsqueeze(-1))
        gf.append((read_i16(hd, 8).float() / 50.0).unsqueeze(-1))
        gf.append((read_i16(hd, 12).float() / self.vp_to_win).unsqueeze(-1))
        cls_tok = self.header_in(torch.cat(gf, -1).to(wdt)).unsqueeze(1)

        return torch.cat([hex_tok, fac_tok, cls_tok], 1)

    def trunk(self, raw: torch.Tensor) -> torch.Tensor:
        x = self.embed(raw)
        bias = self.bias_table.index_select(1, self.buckets.view(-1)).view(
            self.n_heads, self.tokens, self.tokens)
        for blk in self.blocks:
            x = blk(x, bias)
        return x

    def forward(self, raw: torch.Tensor, query_hex: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Returns (logits per action space, value).

        `logits` is laid out to match the engine's action bitmask exactly, so a
        label mask can be compared against it without any per-kind reshaping:
            [0, hexes*6)                movement / cavalry (hex, direction)
            [hexes*6]                   pass
            [0, hexes)                  placement / draft   -- separate head
            [0, hexes*9)                buy                 -- separate head
        Each head keeps its own tensor; the trainer selects by kind.
        """
        x = self.trunk(raw)
        hex_tok = x[:, :self.hexes]
        fac_tok = x[:, self.hexes:self.hexes + self.num_factions]
        cls_tok = x[:, -1]
        q = query_hex.clamp(0, self.hexes - 1).view(-1, 1, 1).expand(-1, 1, self.d)
        query_tok = hex_tok.gather(1, q).squeeze(1)

        b = raw.shape[0]
        return {
            "move": torch.cat([self.head_move(hex_tok).reshape(b, -1), self.head_pass(cls_tok)], 1),
            "hex": self.head_hex(hex_tok).reshape(b, -1),
            "buy": self.head_buy(hex_tok).reshape(b, -1),
            "target": self.head_target(fac_tok).reshape(b, -1),
            "resource": self.head_resource(query_tok),
            "swap": self.head_swap(cls_tok),
            "rect": self.head_rect(query_tok),
            "value": self.head_value(cls_tok).squeeze(-1),
        }


def build_geometry(radius: int, num_factions: int):
    """Neighbour and distance-bucket tables, from cube coordinates.

    Index assignment must match engine_old/geometry.py's cube_hexes_in_radius,
    which HexGrid also uses - every hex index in the encoded record depends on
    it.
    """
    coords = []
    for q in range(-radius, radius + 1):
        r1 = max(-radius, -q - radius)
        r2 = min(radius, -q + radius)
        for r in range(r1, r2 + 1):
            coords.append((q, r, -q - r))
    n = len(coords)
    assert n == hexes_for_radius(radius)
    index = {c: i for i, c in enumerate(coords)}

    directions = [(1, 0, -1), (1, -1, 0), (0, -1, 1), (-1, 0, 1), (-1, 1, 0), (0, 1, -1)]
    nb = torch.empty(n, 7, dtype=torch.long)
    for i, (q, r, s) in enumerate(coords):
        nb[i, 0] = i
        for d, (dq, dr, ds) in enumerate(directions):
            nb[i, 1 + d] = index.get((q + dq, r + dr, s + ds), n)

    def dist(a, b):
        return max(abs(a[0] - b[0]), abs(a[1] - b[1]), abs(a[2] - b[2]))

    tokens = n + num_factions + 1
    bk = torch.empty(tokens, tokens, dtype=torch.long)
    cls = tokens - 1
    for i in range(tokens):
        for j in range(tokens):
            ih, jh = i < n, j < n
            ic, jc = i == cls, j == cls
            if ih and jh:
                b = distance_bucket(dist(coords[i], coords[j]))
            elif ic and jc:
                b = BUCKET_CLS_CLS
            elif ic or jc:
                b = BUCKET_HEX_CLS if (ih or jh) else BUCKET_FACTION_CLS
            elif ih or jh:
                b = BUCKET_HEX_FACTION
            else:
                b = BUCKET_FACTION_FACTION
            bk[i, j] = b
    return nb, bk
