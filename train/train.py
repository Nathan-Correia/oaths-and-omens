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

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data import (KIND_BUY, KIND_HEAD, KIND_NAMES, MULTI_LABEL_KINDS, SampleFile,
                  action_bits, unpack_bits)
from model import BUY_VARIANTS
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
    """One flat batch, plus the kind of each row.

    Deliberately NOT grouped by kind. Grouping meant up to nine separate trunk
    passes of ~28 rows each per batch of 256, and the trunk is essentially all
    of the cost - the heads are a rounding error. Running the trunk once over
    everything and selecting rows per head afterwards is the same computation
    with one big GEMM instead of nine small ones. Measured 3.4x.

    The masks are all the same width on disk, so they stack without padding.
    """
    return {
        "pos": torch.stack([s["pos"] for s in batch]),
        "legal": torch.stack([s["legal"] for s in batch]),
        "chosen": torch.stack([s["chosen"] for s in batch]),
        "won": torch.tensor([s["won"] for s in batch]),
        "ret": torch.tensor([s["ret"] for s in batch]),
        "kind": torch.tensor([s["kind"] for s in batch], dtype=torch.long),
    }


def buy_variant_weights(ds, idx, sample=60000):
    """Per-variant positive weights for the buy BCE.

    The buy label is dominated by two variants - buy_infantry and
    convert_to_archers are ~78 % of everything the teacher picks - while the
    three upgrades are under 2 % between them. Unweighted BCE learns to predict
    "no" for the rare ones and takes the accuracy: measured recall after the
    first run was 11 % for barracks and 0 % for workshop and temple.

    That matters more than the frequency suggests. A barracks lifts the
    per-turn recruit cap on an outpost, and a temple generates victory points
    every round - and victory points are the entire win condition. Weighting by
    inverse square-root frequency lifts the rare classes without letting ten
    temple examples dominate the gradient.
    """
    import numpy as np
    from data import KIND_BUY

    # Vectorised: a Python loop over ten million indices with two numpy scalar
    # lookups each is minutes of pure interpreter overhead.
    sel = np.asarray(idx, dtype=np.int64)
    rows_all = ds.index[sel]
    buy_rows = rows_all[ds.kind[rows_all] == KIND_BUY]
    if buy_rows.size == 0:
        return None
    rng = np.random.default_rng(0)
    if buy_rows.size > sample:
        buy_rows = rng.choice(buy_rows, size=sample, replace=False)

    pos = np.zeros(BUY_VARIANTS, dtype=np.int64)
    legal = np.zeros(BUY_VARIANTS, dtype=np.int64)
    bits = action_bits(KIND_BUY, ds.info.num_hexes, ds.info.num_factions)
    # Slice the mask columns out of the memmap in BULK. Reading 60k samples one
    # at a time through __getitem__ meant 60k scattered reads over a 32 GB file
    # and took longer than an epoch; sorting the row ids first also keeps the
    # access roughly sequential.
    rows = np.sort(buy_rows)
    lo, mid = ds._pos_end, ds._legal_end
    for start in range(0, len(rows), 8192):
        r = rows[start:start + 8192]
        block = np.asarray(ds.mm[r, lo:])
        lg = np.unpackbits(block[:, :ds.info.mask_bytes], axis=1, bitorder="little")[:, :bits]
        ch = np.unpackbits(block[:, ds.info.mask_bytes:], axis=1, bitorder="little")[:, :bits]
        for v in range(BUY_VARIANTS):
            pos[v] += int(ch[:, v::BUY_VARIANTS].sum())
            legal[v] += int(lg[:, v::BUY_VARIANTS].sum())

    rate = np.where(legal > 0, pos / np.maximum(legal, 1), 0.0)
    # Reference is the most common variant, so weights are >= 1 and the common
    # classes keep their current effective learning rate.
    ref = rate.max() if rate.max() > 0 else 1.0
    w = np.where(rate > 0, np.sqrt(ref / np.maximum(rate, 1e-9)), 1.0)
    w = np.clip(w, 1.0, 20.0)
    print("  buy variant weights (rate -> weight):")
    for v in range(BUY_VARIANTS):
        if legal[v]:
            print(f"    variant {v}: chosen {rate[v]*100:5.1f}% of legal -> x{w[v]:.1f}")
    return torch.tensor(w, dtype=torch.float32)


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
    ap.add_argument("--init-from", default="",
                    help="start from this checkpoint instead of random init - "
                         "self-play fine-tuning must, or the policy forgets")
    ap.add_argument("--gamma", type=float, default=0.8,
                    help="discount for the shaped VP return")
    ap.add_argument("--win-bonus", type=float, default=1.0,
                    help="terminal bonus added to the shaped return, so the "
                         "objective still distinguishes scoring from winning")
    ap.add_argument("--reward-scale", type=float, default=0.0,
                    help="VP per unit of shaped reward; 0 auto-calibrates to the "
                         "95th percentile, which is what keeps it off the clip")
    ap.add_argument("--kind-weight", action="append", default=[],
                    metavar="KIND=W",
                    help="scale one decision kind's loss, e.g. --kind-weight buy=8. "
                         "Repeatable. Kinds default to 1.0")
    ap.add_argument("--advantage-beta", type=float, default=0.0,
                    help="SELF-PLAY: advantage-weighted regression. Weights each "
                         "sample by exp((outcome - V(s)) / beta) instead of hard "
                         "filtering. 0 disables. Try 0.5")
    ap.add_argument("--advantage-clip", type=float, default=10.0,
                    help="cap on the advantage weight, so one lucky game cannot "
                         "dominate a batch")
    ap.add_argument("--min-vp-ratio", type=float, default=0.0,
                    help="SELF-PLAY: keep only samples whose seat finished with at "
                         "least this fraction of the winning score. 0 keeps everything; "
                         "0.9 is roughly 'keep what wins'")
    ap.add_argument("--balance-buy", action="store_true", default=True,
                    help="weight rare buy variants (upgrades) in the loss")
    ap.add_argument("--no-balance-buy", dest="balance_buy", action="store_false")
    args = ap.parse_args()

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    print(f"device: {device}")

    kind_weights = [1.0] * len(KIND_NAMES)
    for spec in args.kind_weight:
        name, _, val = spec.partition("=")
        if name not in KIND_NAMES:
            raise SystemExit(f"--kind-weight: unknown kind {name!r}, "
                             f"expected one of {', '.join(KIND_NAMES)}")
        kind_weights[KIND_NAMES.index(name)] = float(val)
    if any(w != 1.0 for w in kind_weights):
        print("kind weights: " + ", ".join(
            f"{n} {w:g}" for n, w in zip(KIND_NAMES, kind_weights) if w != 1.0))

    ds = SampleFile(args.data[0])
    info = ds.info
    ds.shaped_return(gamma=args.gamma, win_bonus=args.win_bonus, scale=args.reward_scale)
    print(f"shaped return: gamma {args.gamma}, win bonus {args.win_bonus}, "
          f"scale {ds.reward_scale:.1f} VP -> mean {ds.ret.mean():.3f} sd {ds.ret.std():.3f}, "
          f"{(ds.ret > 0.01).mean()*100:.0f}% non-zero, "
          f"{(ds.ret >= 0.999).mean()*100:.1f}% at the clip")
    print(f"data: {info.num_records:,} samples, radius {info.radius}, {info.num_factions} factions, "
          f"{info.num_hexes} hexes, {info.record_bytes} B/record")
    print("      by kind: " + ", ".join(f"{k}={v:,}" for k, v in ds.counts_by_kind().items()))
    if info.encoding_version != 1:
        print(f"WARNING: file encoding version {info.encoding_version} != 1; features may be "
              f"misaligned and training will look like it simply does not learn")

    # Self-play filtering. A policy imitating ITSELF learns nothing - the point
    # is to imitate the trajectories that went well, so the corpus is cut down
    # to the seats that scored near the winner's total. This is what makes the
    # iteration a policy IMPROVEMENT step rather than a fixed point (PLAN.md
    # 12.3).
    keep = np.arange(len(ds), dtype=np.int64)
    if args.min_vp_ratio > 0.0:
        rows = ds.index
        vp = ds.final_vp[rows].astype(np.float32)
        mx = ds.max_vp[rows].astype(np.float32)
        # A game nobody scored in says nothing about who played well; drop it
        # rather than letting 0/0 count as a perfect ratio.
        ratio = np.where(mx > 0, vp / np.maximum(mx, 1), 0.0)
        keep = np.nonzero(ratio >= args.min_vp_ratio)[0].astype(np.int64)
        print(f"self-play filter: keeping {len(keep):,} of {len(ds):,} samples "
              f"({len(keep)/max(1,len(ds))*100:.1f}%) with vp >= "
              f"{args.min_vp_ratio:.2f} x winner")
        if len(keep) < 1000:
            raise SystemExit("filter left too few samples; lower --min-vp-ratio")

    n = len(keep)
    if args.limit:
        n = min(n, args.limit)
    n_val = max(1, int(n * args.val_frac))
    n_train = n - n_val
    g = torch.Generator().manual_seed(1234)
    perm = torch.from_numpy(keep)[torch.randperm(len(keep), generator=g)][:n]
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
    if args.init_from:
        blob = torch.load(args.init_from, map_location="cpu", weights_only=False)
        cfg = blob.get("config", {})
        for k, v in (("radius", info.radius), ("num_factions", info.num_factions)):
            if k in cfg and cfg[k] != v:
                raise SystemExit(f"checkpoint {k}={cfg[k]} but data has {v}")
        net.load_state_dict(blob["model"])
        print(f"initialised from {args.init_from} "
              f"(epoch {blob.get('epoch','?')}, val_loss {blob.get('val_loss', float('nan')):.4f})")

    params = sum(p.numel() for p in net.parameters())
    if args.advantage_beta > 0.0:
        print(f"advantage weighting: beta {args.advantage_beta}, clip "
              f"{args.advantage_clip} (no hard filter needed - every sample "
              f"contributes, weighted)")
    print(f"model: d={args.d_model} x{args.blocks} blocks, {params/1000:.0f}k params, "
          f"{net.tokens} tokens")

    buy_w = buy_variant_weights(ds, train_idx) if args.balance_buy else None
    if buy_w is not None:
        buy_w = buy_w.to(device)

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
                pos = groups["pos"].to(device, non_blocking=True)
                kinds = groups["kind"].to(device, non_blocking=True)
                legal_b = groups["legal"].to(device, non_blocking=True)
                chosen_b = groups["chosen"].to(device, non_blocking=True)
                won = groups["won"].to(device, non_blocking=True)
                ret = groups["ret"].to(device, non_blocking=True)
                qh = query_hex_of(pos, info.num_hexes, hex_off).to(device)

                # ONE trunk pass for the whole batch; the heads then read the
                # rows that belong to them.
                outs = net(pos, qh)

                # The value head predicts the SHAPED RETURN, not win/loss.
                # A binary win target at a 12.5% base rate carries far less
                # information per sample than "how much VP follows from here",
                # and the advantage below is only ever as sharp as this
                # baseline is.
                vloss = F.binary_cross_entropy_with_logits(outs["value"], ret)

                # ADVANTAGE WEIGHTING (PLAN.md §12.3).
                #
                # A hard outcome filter asks "did this seat win", which credits
                # every decision in the game equally - including the many a
                # losing seat would have made identically. Measured on the first
                # self-play corpus, winning and losing seats deviated from the
                # policy's own argmax 23.1% and 19.1% of the time: a four-point
                # difference, which is why filtering moved nothing.
                #
                # The value head, by contrast, predicts the eventual winner at
                # AUC 0.928 (0.810 even on turns 0-2). So `outcome - V(s)` is a
                # far sharper per-decision signal than `outcome` alone: it asks
                # "did this go BETTER than the position deserved", which is the
                # part attributable to the actions actually taken.
                #
                # V(s) is detached - this weights the policy loss, it must not
                # push gradients back through the critic.
                adv_w = None
                if args.advantage_beta > 0.0:
                    baseline = torch.sigmoid(outs["value"].detach())
                    adv = ret - baseline
                    adv_w = torch.exp(adv / args.advantage_beta).clamp(
                        max=args.advantage_clip)
                    # Normalised so the weighting changes the SHAPE of the loss,
                    # not its scale - otherwise it silently rescales the learning
                    # rate too.
                    adv_w = adv_w / adv_w.mean().clamp(min=1e-6)

                for kind in torch.unique(kinds).tolist():
                    sel = (kinds == kind).nonzero(as_tuple=True)[0]
                    if sel.numel() == 0:
                        continue
                    bits = action_bits(kind, info.num_hexes, info.num_factions)
                    head = outs[KIND_HEAD[kind]].index_select(0, sel)[:, :bits]
                    legal = unpack_bits(legal_b.index_select(0, sel), bits)
                    chosen = unpack_bits(chosen_b.index_select(0, sel), bits)

                    # -inf on illegal actions, so they can never win an argmax
                    # and contribute nothing to the softmax.
                    masked = head.masked_fill(legal < 0.5, float("-inf"))
                    if kind in MULTI_LABEL_KINDS:
                        # Set-valued: independent Bernoulli per legal action.
                        w = legal
                        if kind == KIND_BUY and buy_w is not None:
                            # Upweight a POSITIVE example of a rare variant. Only
                            # the positives: upweighting its negatives too would
                            # just make the class louder, not more likely.
                            tile = buy_w.repeat(head.shape[1] // BUY_VARIANTS)
                            w = legal * (1.0 + (tile - 1.0) * chosen)
                        if adv_w is not None:
                            w = w * adv_w.index_select(0, sel).unsqueeze(1)
                        loss = F.binary_cross_entropy_with_logits(
                            head, chosen, weight=w, reduction="sum") / w.sum().clamp(min=1)
                        correct = ((head > 0).float() * legal == chosen * legal).all(1).sum()
                    else:
                        tgt = chosen.argmax(1)
                        if adv_w is None:
                            loss = F.cross_entropy(masked, tgt)
                        else:
                            per = F.cross_entropy(masked, tgt, reduction="none")
                            w = adv_w.index_select(0, sel)
                            loss = (per * w).sum() / w.sum().clamp(min=1e-6)
                        correct = (masked.argmax(1) == tgt).sum()

                    k = sel.numel()
                    # PER-KIND LOSS WEIGHT.
                    #
                    # Every kind contributes in proportion to its row count,
                    # which is the right default and the wrong one here. Buy is
                    # 12.25% of rows and, measured by the fallback ablation,
                    # 100% of the strength gap: handing buy to tactician takes
                    # the policy from 39.4% to 48.2% against it, while movement,
                    # cavalry, target and rectification each move it by nothing.
                    # Gradient share should follow what a head is worth, not how
                    # often the engine happens to ask about it.
                    batch_loss = batch_loss + loss * k * kind_weights[kind]
                    batch_n += k
                    st = per_kind[kind]
                    st[0] += float(loss.detach()) * k
                    st[1] += int(correct)
                    st[2] += k
                batch_loss = batch_loss + args.value_weight * vloss * max(1, batch_n)
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
