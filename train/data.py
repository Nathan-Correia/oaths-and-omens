"""Reader for the training files oo_dump_training writes.

Format is defined in engine/include/oo/train_data.hpp. Fixed-size records after
a 32-byte header, so the whole file is memory-mapped and a sample is a slice -
no parsing, and shuffling is just permuting indices.
"""

import struct
from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset

MAGIC = 0x4F4F5431
SAMPLE_HEADER_BYTES = 16
FILE_HEADER_BYTES = 32

# DecisionKind, mirroring engine/include/oo/policy.hpp.
KIND_BUY, KIND_MOVEMENT, KIND_CAVALRY, KIND_TARGET, KIND_RECT, \
    KIND_RESOURCE, KIND_PLACEMENT, KIND_DRAFT, KIND_SWAP = range(9)

KIND_NAMES = ["buy", "movement", "cavalry", "target", "rectification",
              "resource", "placement", "draft", "swap"]

# Which head each kind is scored by, and how wide its action space is.
KIND_HEAD = {
    KIND_MOVEMENT: "move", KIND_CAVALRY: "move",
    KIND_PLACEMENT: "hex", KIND_DRAFT: "hex",
    KIND_BUY: "buy", KIND_TARGET: "target",
    KIND_RESOURCE: "resource", KIND_SWAP: "swap", KIND_RECT: "rect",
}

# Kinds whose label is a SET rather than one choice - they need a multi-label
# loss, not cross-entropy (see train.py).
MULTI_LABEL_KINDS = {KIND_BUY, KIND_RECT}


@dataclass
class FileInfo:
    radius: int
    num_factions: int
    num_hexes: int
    mask_bytes: int
    record_bytes: int
    position_bytes: int
    num_records: int
    encoding_version: int


def read_info(path) -> FileInfo:
    with open(path, "rb") as f:
        raw = f.read(FILE_HEADER_BYTES)
    (magic, version, enc_version, radius, num_factions, num_hexes, mask_bytes,
     record_bytes, position_bytes, num_records) = struct.unpack("<IHHHHHHIIQ", raw)
    if magic != MAGIC:
        raise ValueError(f"{path} is not a training file (magic {magic:#x})")
    if version != 1:
        raise ValueError(f"{path} is version {version}, expected 1")
    return FileInfo(radius, num_factions, num_hexes, mask_bytes, record_bytes,
                    position_bytes, num_records, enc_version)


def action_bits(kind: int, num_hexes: int, num_factions: int) -> int:
    """Mirrors oo::action_bits. Bits past this are zero in both masks."""
    if kind in (KIND_MOVEMENT, KIND_CAVALRY):
        return num_hexes * 6 + 1
    if kind == KIND_BUY:
        return num_hexes * 9
    if kind in (KIND_PLACEMENT, KIND_DRAFT):
        return num_hexes
    if kind == KIND_TARGET:
        return num_factions
    if kind in (KIND_RESOURCE, KIND_SWAP):
        return 2
    if kind == KIND_RECT:
        return 16 * 3
    return 0


class SampleFile(Dataset):
    """One dumped file. Records are fixed size, so indexing is arithmetic."""

    def __init__(self, path, kinds=None):
        self.path = str(path)
        self.info = read_info(path)
        self.mm = np.memmap(self.path, dtype=np.uint8, mode="r", offset=FILE_HEADER_BYTES,
                            shape=(self.info.num_records, self.info.record_bytes))

        # The header is the first 16 bytes of every record, so kind and outcome
        # can be read for the whole file in one vectorised pass - filtering by
        # kind never touches the position payload.
        heads = self.mm[:, :SAMPLE_HEADER_BYTES]
        self.kind = np.asarray(heads[:, 0])
        self.faction = np.asarray(heads[:, 3].astype(np.int8))
        self.final_vp = np.asarray(heads[:, 4:6]).copy().view(np.int16).reshape(-1)
        self.max_vp = np.asarray(heads[:, 6:8]).copy().view(np.int16).reshape(-1)
        self.won = np.asarray(heads[:, 2])

        if kinds is None:
            self.index = np.arange(self.info.num_records, dtype=np.int64)
        else:
            keep = np.isin(self.kind, list(kinds))
            self.index = np.nonzero(keep)[0].astype(np.int64)

        self._pos_end = SAMPLE_HEADER_BYTES + self.info.position_bytes
        self._legal_end = self._pos_end + self.info.mask_bytes

    def __len__(self):
        return len(self.index)

    def counts_by_kind(self):
        return {KIND_NAMES[k]: int((self.kind == k).sum()) for k in range(9)
                if (self.kind == k).any()}

    def __getitem__(self, i):
        r = int(self.index[i])
        rec = self.mm[r]
        pos = torch.from_numpy(np.asarray(rec[SAMPLE_HEADER_BYTES:self._pos_end]).copy())
        legal = torch.from_numpy(np.asarray(rec[self._pos_end:self._legal_end]).copy())
        chosen = torch.from_numpy(np.asarray(rec[self._legal_end:]).copy())
        # Value target: did this sample's own seat win. Scaled VP is available
        # too but win/loss is the better-conditioned signal to start from.
        return {
            "pos": pos,
            "legal": legal,
            "chosen": chosen,
            "kind": int(self.kind[r]),
            "won": float(self.won[r]),
            "vp": float(self.final_vp[r]) / 50.0,
        }


def unpack_bits(mask_bytes: torch.Tensor, n_bits: int) -> torch.Tensor:
    """[B, mask_bytes] uint8 -> [B, n_bits] float, little-endian within a byte.

    Matches oo::set_bit / oo::get_bit: bit i lives in byte i>>3 at 1<<(i&7).
    """
    b = mask_bytes.shape[0]
    bits = torch.arange(8, device=mask_bytes.device, dtype=torch.uint8)
    out = (mask_bytes.unsqueeze(-1) >> bits) & 1
    return out.view(b, -1)[:, :n_bits].float()
