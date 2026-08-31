#!/usr/bin/env python3
"""establish and check the classical baselines.

Three numbers come out of this script. Everything later is measured
against them.

  1. bypassed MSG      -- the loop with no suppression
  2. oracle notch bank -- K slots placed on the *measured* howl modes
  3. greedy cascade    -- re-detect after each slot, i.e. the
                          self-allocating behaviour of ic.notchBank

The learned policy has to beat (2). If it only beats (3) it has merely
rediscovered oracle placement, which is not interesting.

Note on the FAUST reference: faust-icc measures +12 dB of added stable
gain using IIR notches (fi.notchw). This port uses short FIR notches,
which carry more group delay inside the loop, so the ceiling here is
lower. The gap is expected and documented in docs/validation.md -- do not
"fix" it by loosening the howling criterion.

Run:
    python scripts/01_verify_plant.py
    python scripts/01_verify_plant.py --speech-dir ~/data/LibriSpeech
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from stabilitygan.data import load_speech_files, speech_batch
from stabilitygan.msg import measure_msg, probe_howl_frequencies
from stabilitygan.notch import NotchBank
from stabilitygan.plant import CabinConfig, CabinPath, ClosedLoop


def bank_suppressor(bank, freqs_hz, sample_rate):
    f = torch.tensor(np.asarray(freqs_hz, dtype=np.float32) / (sample_rate / 2.0))
    f = f.clamp(0.02, 0.98).unsqueeze(0)
    return bank.as_suppressor(f, torch.ones(1, f.shape[1]))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--speech-dir", default=None)
    ap.add_argument("--sample-rate", type=int, default=16000)
    ap.add_argument("--seconds", type=float, default=1.0)
    ap.add_argument("--frame", type=int, default=256)
    ap.add_argument("--slots", type=int, default=4)
    ap.add_argument("--filter-width", type=int, default=31)
    ap.add_argument("--notch-width", type=float, default=0.08)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min-added-db", type=float, default=5.0)
    ap.add_argument("--out", default="docs/baselines.json")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    sr = args.sample_rate

    loop = ClosedLoop(CabinPath(CabinConfig(sample_rate=sr)), frame_size=args.frame)
    n = int(args.seconds * sr)
    n -= n % args.frame

    files = load_speech_files(args.speech_dir)
    print(f"speech source : {('%d files' % len(files)) if files else 'synthetic'}")
    speech = speech_batch(1, n, sr, files, rng)

    # ---- 1. bypassed ---------------------------------------------------
    msg_bypass = float(measure_msg(loop, speech, None, sr))
    print(f"\n[1] bypassed MSG           : {msg_bypass:+6.2f} dB")

    # ---- 2. oracle placement -------------------------------------------
    hz = probe_howl_frequencies(loop, speech, msg_bypass + 6.0, args.slots, sr)
    if len(hz) == 0:
        print("no howling modes found -- the loop never reaches instability")
        return 1
    hz = np.pad(hz, (0, args.slots - len(hz)), mode="edge")
    bank = NotchBank(args.slots, args.filter_width, args.notch_width)
    msg_oracle = float(measure_msg(loop, speech, bank_suppressor(bank, hz, sr), sr))
    print(f"    howl modes (Hz)       : {np.round(hz, 1).tolist()}")
    print(f"[2] oracle notch bank MSG : {msg_oracle:+6.2f} dB"
          f"   (added {msg_oracle - msg_bypass:+.2f} dB)")

    # ---- 3. greedy self-allocating cascade -----------------------------
    chosen: list[float] = []
    msg_greedy = msg_bypass
    for _ in range(args.slots):
        supp = (
            bank_suppressor(
                NotchBank(len(chosen), args.filter_width, args.notch_width), chosen, sr
            )
            if chosen
            else None
        )
        cur = float(measure_msg(loop, speech, supp, sr))
        nxt = probe_howl_frequencies(loop, speech, cur + 6.0, 1, sr, suppressor=supp)
        if len(nxt) == 0:
            break
        chosen.append(float(nxt[0]))
        msg_greedy = float(
            measure_msg(
                loop,
                speech,
                bank_suppressor(
                    NotchBank(len(chosen), args.filter_width, args.notch_width), chosen, sr
                ),
                sr,
            )
        )
    print(f"[3] greedy cascade MSG    : {msg_greedy:+6.2f} dB"
          f"   (added {msg_greedy - msg_bypass:+.2f} dB)")
    print(f"    greedy picks (Hz)     : {np.round(chosen, 1).tolist()}")

    added = msg_oracle - msg_bypass
    print("\n" + "=" * 56)
    print(f"  target for the learned policy: beat {added:+.2f} dB")
    print("=" * 56)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "sample_rate": sr,
                "frame": args.frame,
                "slots": args.slots,
                "filter_width": args.filter_width,
                "notch_width": args.notch_width,
                "msg_bypass_db": msg_bypass,
                "msg_oracle_db": msg_oracle,
                "msg_greedy_db": msg_greedy,
                "added_oracle_db": added,
                "added_greedy_db": msg_greedy - msg_bypass,
                "howl_modes_hz": np.round(hz, 2).tolist(),
            },
            indent=2,
        )
    )
    print(f"\nwrote {out}")

    if added >= args.min_added_db:
        print("GATE PASSED")
        return 0
    print(
        f"GATE FAILED -- oracle bank bought only {added:+.2f} dB "
        f"(expected >= {args.min_added_db:.1f}). Do not proceed."
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
