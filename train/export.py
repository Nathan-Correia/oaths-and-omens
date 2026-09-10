"""Exports a policy to TorchScript for the C++ engine (PLAN.md §10, §12.4).

    python train/export.py --init  --out checkpoints/init.ts   --radius 7 --factions 8
    python train/export.py --ckpt checkpoints/bc/best.pt --out checkpoints/bc.ts

model.py is the single definition of the network. The engine loads the scripted
module rather than rebuilding the architecture in C++, so there is no second
definition to drift - which was a real hazard, because a feature-layout mismatch
does not raise, it silently feeds the net misaligned inputs.

`--init` writes an untrained module, which is what `oo_run --neural` needs for a
speed benchmark before any training has happened.

The scripted module carries `radius`, `num_factions` and `encoding_version` as
attributes so the engine can refuse a checkpoint that does not match the board it
is about to play, instead of quietly producing nonsense.
"""

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model import Policy


def build(radius, factions, d_model, blocks, heads, state=None):
    net = Policy(radius=radius, num_factions=factions, d_model=d_model,
                 n_blocks=blocks, n_heads=heads)
    if state is not None:
        missing, unexpected = net.load_state_dict(state, strict=False)
        if missing:
            raise SystemExit(f"checkpoint is missing {len(missing)} tensors, first: {missing[:3]}")
        if unexpected:
            raise SystemExit(f"checkpoint has {len(unexpected)} unexpected tensors, "
                             f"first: {unexpected[:3]}")
    net.eval()
    return net


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--ckpt", help="a .pt from train.py; omit with --init")
    ap.add_argument("--init", action="store_true", help="export untrained weights")
    ap.add_argument("--radius", type=int, default=7)
    ap.add_argument("--factions", type=int, default=8)
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--blocks", type=int, default=4)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--half", action="store_true",
                    help="export fp16 weights (what the engine wants on a GPU)")
    args = ap.parse_args()

    if not args.init and not args.ckpt:
        raise SystemExit("give --ckpt PATH or --init")

    if args.ckpt:
        blob = torch.load(args.ckpt, map_location="cpu", weights_only=False)
        cfg = blob.get("config", {})
        # The checkpoint's own config wins over the flags - exporting a trained
        # net at the wrong width would fail on load with an unhelpful shape error.
        radius = cfg.get("radius", args.radius)
        factions = cfg.get("num_factions", args.factions)
        d_model = cfg.get("d_model", args.d_model)
        blocks = cfg.get("n_blocks", args.blocks)
        heads = cfg.get("n_heads", args.heads)
        net = build(radius, factions, d_model, blocks, heads, blob["model"])
        print(f"loaded {args.ckpt} (epoch {blob.get('epoch', '?')}, "
              f"val_loss {blob.get('val_loss', float('nan')):.4f})")
    else:
        torch.manual_seed(args.seed)
        radius, factions = args.radius, args.factions
        d_model, blocks, heads = args.d_model, args.blocks, args.heads
        net = build(radius, factions, d_model, blocks, heads)
        print("exporting UNTRAINED weights")

    if args.half:
        # Done HERE and not in C++: Module::to(kFloat16) on the engine side
        # converts every tensor including the integer geometry buffers, and a
        # half index tensor fails inside gather with an unhelpful message.
        # nn.Module.half() only touches floating-point tensors.
        net.half()

    scripted = torch.jit.script(net)
    # Attributes the engine reads back to validate the checkpoint against the
    # board it is about to play.
    scripted.radius = radius
    scripted.num_factions = factions
    scripted.encoding_version = 1

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    scripted.save(str(out))

    params = sum(p.numel() for p in net.parameters())
    print(f"wrote {out} - radius {radius}, {factions} factions, d={d_model} x{blocks} blocks, "
          f"{params/1000:.0f}k params")

    # Round-trip it, so a module that saves but cannot be loaded fails here
    # rather than inside the engine.
    reloaded = torch.jit.load(str(out))
    if args.half:
        raw_dtype = next(reloaded.parameters()).dtype
        print(f"weights are {raw_dtype}")
    hexes = 3 * radius * (radius + 1) + 1
    raw = torch.zeros(2, 16 + 24 * 10 + 12 * 331, dtype=torch.uint8)
    raw[:, 6] = hexes & 0xFF
    raw[:, 7] = hexes >> 8
    out_heads = reloaded(raw, torch.zeros(2, dtype=torch.long))
    print("round-trip ok; heads: " +
          ", ".join(f"{k}{tuple(v.shape)}" for k, v in sorted(out_heads.items())))


if __name__ == "__main__":
    main()
