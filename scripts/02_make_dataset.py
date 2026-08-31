#!/usr/bin/env python3
"""generate (loop recording, MSG label) pairs.

This is the expensive step: every label needs a full gain sweep. Run it
once on the CUDA box and cache the result.

    python scripts/02_make_dataset.py --n 2000 --out data/train.pt
    python scripts/02_make_dataset.py --n 300 --seed 99 --out data/val.pt
"""
from __future__ import annotations
import argparse, sys, time
from pathlib import Path
import numpy as np, torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from stabilitygan.data import load_speech_files, speech_batch
from stabilitygan.msg import measure_msg
from stabilitygan.plant import CabinConfig, CabinPath, ClosedLoop


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--speech-dir", default=None)
    ap.add_argument("--sample-rate", type=int, default=16000)
    ap.add_argument("--seconds", type=float, default=1.0)
    ap.add_argument("--frame", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="data/train.pt")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    sr, dev = args.sample_rate, torch.device(args.device)
    n = int(args.seconds * sr); n -= n % args.frame
    files = load_speech_files(args.speech_dir)
    print(f"speech: {('%d files' % len(files)) if files else 'synthetic'}   device: {dev}")

    recs, labels, metas = [], [], []
    t0 = time.time()
    for i in range(args.n):
        cfg = CabinConfig.random(rng, sr)
        loop = ClosedLoop(CabinPath(cfg), frame_size=args.frame).to(dev)
        speech = speech_batch(1, n, sr, files, rng).to(dev)
        msg = float(measure_msg(loop, speech, None, sr))

        # record the loop running at a gain sampled around MSG, so the net
        # sees both comfortably-stable and about-to-howl examples
        g = msg + float(rng.uniform(-12.0, 4.0))
        mic, _ = loop(speech, g, suppressor=None)

        recs.append(mic.squeeze(0).cpu())
        labels.append(msg - g)   # headroom in dB: how much more gain is safe
        metas.append([cfg.delay_ms, cfg.res_freq, cfg.res_gain_db, cfg.res_q, cfg.hf_cut_hz, g])

        if (i + 1) % 25 == 0:
            el = time.time() - t0
            print(f"  {i+1}/{args.n}  {el:.0f}s  eta {el/(i+1)*(args.n-i-1):.0f}s", flush=True)

    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "recordings": torch.stack(recs),
        "headroom_db": torch.tensor(labels, dtype=torch.float32),
        "meta": torch.tensor(metas, dtype=torch.float32),
        "sample_rate": sr, "frame": args.frame,
    }, out)
    print(f"\nwrote {out}  ({len(recs)} examples, "
          f"headroom {np.mean(labels):+.1f} +/- {np.std(labels):.1f} dB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
