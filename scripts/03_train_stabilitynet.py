#!/usr/bin/env python3
"""train the differentiable MSG surrogate.

StabilityNet predicts remaining gain headroom in dB from a recording of
the loop. It is the differentiable stand-in for msg.measure_msg, exactly
as MetricGAN's discriminator stands in for PESQ.


    python scripts/03_train_stabilitynet.py --train data/train.pt --val data/val.pt
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import torch
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from stabilitygan.models import StabilityNet

HEADROOM_MIN, HEADROOM_MAX = -20.0, 20.0


def to_unit(db): return ((db - HEADROOM_MIN) / (HEADROOM_MAX - HEADROOM_MIN)).clamp(0, 1)
def to_db(u):    return u * (HEADROOM_MAX - HEADROOM_MIN) + HEADROOM_MIN


def load(path):
    d = torch.load(path, weights_only=False)
    return TensorDataset(d["recordings"], d["headroom_db"]), d["sample_rate"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="data/train.pt")
    ap.add_argument("--val", default="data/val.pt")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--max-mae", type=float, default=3.0)
    ap.add_argument("--out", default="checkpoints/stabilitynet.pt")
    args = ap.parse_args()

    dev = torch.device(args.device)
    tr, sr = load(args.train)
    va, _ = load(args.val)
    dl_tr = DataLoader(tr, batch_size=args.batch, shuffle=True, drop_last=True)
    dl_va = DataLoader(va, batch_size=args.batch)

    net = StabilityNet().to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    best = float("inf")

    for ep in range(1, args.epochs + 1):
        net.train()
        for wav, db in dl_tr:
            wav, db = wav.to(dev), db.to(dev)
            loss = torch.nn.functional.mse_loss(net(wav), to_unit(db))
            opt.zero_grad(); loss.backward(); opt.step()

        net.eval(); err = []
        with torch.no_grad():
            for wav, db in dl_va:
                pred = to_db(net(wav.to(dev))).cpu()
                err.append((pred - db).abs())
        mae = float(torch.cat(err).mean())
        print(f"epoch {ep:3d}  val MAE {mae:6.2f} dB")
        if mae < best:
            best = mae
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            torch.save({"state_dict": net.state_dict(), "val_mae_db": mae,
                        "sample_rate": sr}, args.out)

    print(f"\nbest val MAE: {best:.2f} dB  ->  {args.out}")
    if best <= args.max_mae:
        print("GATE PASSED -- surrogate is good enough to train a policy against")
        return 0
    print(f"GATE FAILED -- MAE {best:.2f} dB exceeds {args.max_mae:.1f} dB.\n"
          "More data, longer clips, or a bigger net. Do NOT start step 04.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
