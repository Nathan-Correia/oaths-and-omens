"""Behaviour cloning from a scripted teacher (PLAN.md §12.2, M10b).

    python train/train.py --data data/r7f8.bin --epochs 4 --out checkpoints/bc

Trains the §10 policy to reproduce the teacher's decisions. Nine decision kinds
share one trunk, so a batch is grouped by kind and each group is scored by its
own head - which is why the sampler keeps kinds together rather than shuffling
them uniformly.

TWO LOSSES, because the labels are two different shapes:
  - Single-choice kinds (movement, cavalry, placement, draft, target, resource,
    swap) use masked cross-entropy over the legal set.
  - Set-valued kinds (buy, rectification) use masked BCE. `decide_buy` picks a
    SUBSET of legal actions and rectification a multiset, so cross-entropy would
    be asserting there is one right answer when there is not.

Plus an auxiliary value head on whether this seat won (§10.6). It costs almost
nothing and reliably improves the trunk, and search will need it later.

A NOTE ON THE ACCURACY NUMBERS. Top-1 agreement understates a good policy here.
Tactician breaks ties by taking the first maximum in hex/direction index order,
so on any position with several equally-scored moves the label is an arbitrary
pick among them (§12.4). Judge with the evaluation harness, not this number.
"""

import argparse
import json
import math
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data import (KIND_HEAD, KIND_NAMES, MULTI_LABEL_KINDS, SampleFile, action_bits,
                  unpack_bits)
from model import Policy


def fmt_time(s):
    if s < 90:
        return f"{s:.0f}s"
    if s < 5400:
        return f"{s / 60:.1f}m"
    return f"{s / 3600:.1f}h"


class Bar:
    """A single-line progress bar that does not need tqdm.

    Deliberately dependency-free: this repo has no Python requirements at all
    (§1.2 deleted them), and adding one just to draw a bar would be the first
    step back down that road.
    """

    def __init__(self, total, label, width=32):
        self.total = max(1, total)
        self.label = label
        self.width = width
        self.start = time.time()
        self.n = 0
        self._last = 0.0

    def update(self, n=1, suffix=""):
        self.n += n
        now = time.time()
        if now - self._last < 0.1 and self.n < self.total:
            return
        self._last = now
        frac = min(1.0, self.n / self.total)
        filled = int(frac * self.width)
        bar = "=" * filled + " " * (self.width - filled)
        el = now - self.start
        rate = self.n / el if el > 0 else 0
        eta = (self.total - self.n) / rate if rate > 0 else 0
        sys.stdout.write(
            f"\r{self.label} [{bar}] {self.n}/{self.total} "
            f"{rate:.0f}/s eta {fmt_time(eta)} {suffix}   ")
        sys.stdout.flush()

    def close(self, suffix=""):
        el = time.time() - self.start
        bar = "=" * self.width
        sys.stdout.write(f"\r{self.label} [{bar}] {self.n}/{self.total} in {fmt_time(el)} {suffix}"
                         + " " * 20 + "\n")
        sys.stdout.flush()


def collate(batch):
    """Groups a batch by decision kind - one head runs per group."""
    by_kind = defaultdict(list)
    for s in batch:
        by_kind[s["kind"]].append(s)
    out = {}
    for kind, items in by_kind.items():
        out[kind] = {
            "pos": torch.stack([s["pos"] for s in items]),
            "legal": torch.stack([s["legal"] for s in items]),
            "chosen": torch.stack([s["chosen"] for s in items]),
            "won": torch.tensor([s["won"] for s in items]),
        }
    return out


def query_hex_of(pos: torch.Tensor, num_hexes: int, hex_off: int) -> torch.Tensor:
    """Recovers the query hex from the encoded record's kHexQuery flag (bit 4).

    The engine marks it in the position rather than passing it separately, so
    the trainer reads it back the same way inference does - one less thing that
    can disagree between the two.
    """
    flags = pos[:, hex_off:hex_off + 12 * num_hexes].view(-1, num_hexes, 12)[:, :, 7]
    is_query = ((flags >> 4) & 1).to(torch.int64)
    # argmax gives the first set bit, or 0 when none is set, which matches the
    # C++ decoder's fallback.
    return is_query.argmax(dim=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, nargs="+", help="training file(s) from oo_dump_training")
    ap.add_argument("--out", default="checkpoints/bc", help="checkpoint directory")
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--blocks", type=int, default=4)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--value-weight", type=float, default=0.5)
    ap.add_argument("--val-frac", type=float, default=0.02)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="cap samples, for a quick run")
    args = ap.parse_args()

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    print(f"device: {device}")

    ds = SampleFile(args.data[0])
    info = ds.info
    print(f"data: {info.num_records:,} samples, radius {info.radius}, {info.num_factions} factions, "
          f"{info.num_hexes} hexes, {info.record_bytes} B/record")
    print("      by kind: " + ", ".join(f"{k}={v:,}" for k, v in ds.counts_by_kind().items()))
    if info.encoding_version != 1:
        print(f"WARNING: file encoding version {info.encoding_version} != 1; features may be "
              f"misaligned and training will look like it simply does not learn")

    n = len(ds)
    if args.limit:
        n = min(n, args.limit)
    n_val = max(1, int(n * args.val_frac))
    n_train = n - n_val
    g = torch.Generator().manual_seed(1234)
    perm = torch.randperm(len(ds), generator=g)[:n]
    train_idx, val_idx = perm[:n_train].tolist(), perm[n_train:].tolist()
    print(f"split: {n_train:,} train / {n_val:,} val")

    train_ds = torch.utils.data.Subset(ds, train_idx)
    val_ds = torch.utils.data.Subset(ds, val_idx)
    dl_kw = dict(batch_size=args.batch, collate_fn=collate, num_workers=args.workers,
                 pin_memory=(device.type == "cuda"), persistent_workers=args.workers > 0)
    train_dl = DataLoader(train_ds, shuffle=True, drop_last=True, **dl_kw)
    val_dl = DataLoader(val_ds, shuffle=False, **dl_kw)

    net = Policy(radius=info.radius, num_factions=info.num_factions, d_model=args.d_model,
                 n_blocks=args.blocks, n_heads=args.heads).to(device)
    params = sum(p.numel() for p in net.parameters())
    print(f"model: d={args.d_model} x{args.blocks} blocks, {params/1000:.0f}k params, "
          f"{net.tokens} tokens")

    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=0.01)
    steps = args.epochs * max(1, len(train_dl))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=steps,
                                                pct_start=0.1)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    hex_off = 16 + 24 * 10

    def run_split(dl, train: bool, label: str):
        net.train(train)
        tot_loss, tot_n = 0.0, 0
        per_kind = defaultdict(lambda: [0.0, 0, 0])  # loss, correct, count
        bar = Bar(len(dl), label)
        for groups in dl:
            if train:
                opt.zero_grad(set_to_none=True)
            batch_loss = 0.0
            batch_n = 0
            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                for kind, g in groups.items():
                    pos = g["pos"].to(device, non_blocking=True)
                    qh = query_hex_of(pos, info.num_hexes, hex_off).to(device)
                    outs = net(pos, qh)
                    head = outs[KIND_HEAD[kind]]
                    bits = action_bits(kind, info.num_hexes, info.num_factions)
                    head = head[:, :bits]
                    legal = unpack_bits(g["legal"].to(device), bits)
                    chosen = unpack_bits(g["chosen"].to(device), bits)

                    # -inf on illegal actions, so they can never win an argmax
                    # and contribute nothing to the softmax.
                    masked = head.masked_fill(legal < 0.5, float("-inf"))
                    if kind in MULTI_LABEL_KINDS:
                        # Set-valued: independent Bernoulli per legal action.
                        loss = F.binary_cross_entropy_with_logits(
                            head, chosen, weight=legal, reduction="sum") / max(1, legal.sum())
                        correct = ((head > 0).float() * legal == chosen * legal).all(1).sum()
                    else:
                        tgt = chosen.argmax(1)
                        loss = F.cross_entropy(masked, tgt)
                        correct = (masked.argmax(1) == tgt).sum()

                    vloss = F.binary_cross_entropy_with_logits(
                        outs["value"], g["won"].to(device))
                    total = loss + args.value_weight * vloss
                    k = pos.shape[0]
                    batch_loss = batch_loss + total * k
                    batch_n += k
                    st = per_kind[kind]
                    st[0] += float(loss.detach()) * k
                    st[1] += int(correct)
                    st[2] += k
            if batch_n == 0:
                bar.update()
                continue
            loss = batch_loss / batch_n
            if train:
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
                sched.step()
            tot_loss += float(loss.detach()) * batch_n
            tot_n += batch_n
            bar.update(suffix=f"loss {tot_loss / max(1, tot_n):.4f}")
        bar.close(suffix=f"loss {tot_loss / max(1, tot_n):.4f}")
        return tot_loss / max(1, tot_n), per_kind

    history = []
    best = float("inf")
    for epoch in range(1, args.epochs + 1):
        print(f"\nepoch {epoch}/{args.epochs}  lr {sched.get_last_lr()[0]:.2e}")
        tr_loss, _ = run_split(train_dl, True, "  train")
        with torch.no_grad():
            va_loss, va_kind = run_split(val_dl, False, "  val  ")

        print("  top-1 agreement with the teacher (understates - see the module docstring):")
        for kind, (lsum, corr, cnt) in sorted(va_kind.items()):
            if cnt:
                print(f"    {KIND_NAMES[kind]:<14} {corr/cnt*100:5.1f}%  ({cnt:,} samples)")

        history.append({"epoch": epoch, "train_loss": tr_loss, "val_loss": va_loss})
        torch.save({"model": net.state_dict(),
                    "config": {"radius": info.radius, "num_factions": info.num_factions,
                               "d_model": args.d_model, "n_blocks": args.blocks,
                               "n_heads": args.heads, "encoding_version": 1},
                    "epoch": epoch, "val_loss": va_loss},
                   out_dir / "last.pt")
        if va_loss < best:
            best = va_loss
            torch.save({"model": net.state_dict(),
                        "config": {"radius": info.radius, "num_factions": info.num_factions,
                                   "d_model": args.d_model, "n_blocks": args.blocks,
                                   "n_heads": args.heads, "encoding_version": 1},
                        "epoch": epoch, "val_loss": va_loss},
                       out_dir / "best.pt")
            print(f"  new best ({va_loss:.4f}) -> {out_dir/'best.pt'}")
        (out_dir / "history.json").write_text(json.dumps(history, indent=2))

    print(f"\ndone. best val loss {best:.4f}, checkpoints in {out_dir}")
    print("Next: the evaluation harness (M10a) - the loss above measures agreement with")
    print("tactician's arbitrary tie-breaks, not strength.")


if __name__ == "__main__":
    main()
