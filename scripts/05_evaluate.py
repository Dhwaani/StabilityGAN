#!/usr/bin/env python3
"""evaluate the learned policy.

Runs the real (non-differentiable) MSG sweep on held-out cabins for:
  bypassed / greedy cascade / oracle placement / learned policy
and writes a table plus the audio renders and a summary plot.

    python scripts/05_evaluate.py --policy checkpoints/policy.pt --n 30
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np, torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from stabilitygan.data import load_speech_files, speech_batch
from stabilitygan.models import NotchPolicy
from stabilitygan.msg import measure_msg, probe_howl_frequencies
from stabilitygan.notch import NotchBank
from stabilitygan.plant import CabinConfig, CabinPath, ClosedLoop
from stabilitygan.faust_export import export_preset


def bank_supp(bank, hz, sr):
    f = torch.tensor(np.asarray(hz, dtype=np.float32) / (sr / 2)).clamp(0.02, 0.98).unsqueeze(0)
    return bank.as_suppressor(f, torch.ones(1, f.shape[1]))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", default="checkpoints/policy.pt")
    ap.add_argument("--speech-dir", default=None)
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--seconds", type=float, default=1.0)
    ap.add_argument("--frame", type=int, default=256)
    ap.add_argument("--sample-rate", type=int, default=16000)
    ap.add_argument("--audio-out", default="audio")
    ap.add_argument("--out", default="docs/results.json")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed); torch.manual_seed(args.seed)
    sr = args.sample_rate
    n = int(args.seconds * sr); n -= n % args.frame

    ck = torch.load(args.policy, weights_only=False, map_location="cpu")
    policy = NotchPolicy(n_slots=ck["slots"]); policy.load_state_dict(ck["state_dict"]); policy.eval()
    bank = NotchBank(ck["slots"], ck["filter_width"], ck["notch_width"])
    files = load_speech_files(args.speech_dir)

    rows = []
    for i in range(args.n):
        cfg = CabinConfig.random(rng, sr)
        loop = ClosedLoop(CabinPath(cfg), frame_size=args.frame)
        speech = speech_batch(1, n, sr, files, rng)

        m_by = float(measure_msg(loop, speech, None, sr))

        hz = probe_howl_frequencies(loop, speech, m_by + 6.0, ck["slots"], sr)
        if len(hz) == 0:
            continue
        hz = np.pad(hz, (0, ck["slots"] - len(hz)), mode="edge")
        m_or = float(measure_msg(loop, speech, bank_supp(bank, hz, sr), sr))

        chosen: list[float] = []
        m_gr = m_by
        for _ in range(ck["slots"]):
            s = bank_supp(NotchBank(len(chosen), ck["filter_width"], ck["notch_width"]),
                          chosen, sr) if chosen else None
            cur = float(measure_msg(loop, speech, s, sr))
            nx = probe_howl_frequencies(loop, speech, cur + 6.0, 1, sr, suppressor=s)
            if len(nx) == 0:
                break
            chosen.append(float(nx[0]))
            m_gr = float(measure_msg(loop, speech,
                                     bank_supp(NotchBank(len(chosen), ck["filter_width"],
                                                         ck["notch_width"]), chosen, sr), sr))

        with torch.no_grad():
            probe, _ = loop(speech, m_by + 8.0, suppressor=None)
            f, d = policy(probe)
        m_lp = float(measure_msg(loop, speech, bank.as_suppressor(f, d), sr))

        rows.append({"bypass": m_by, "greedy": m_gr, "oracle": m_or, "learned": m_lp,
                     "cabin": [cfg.delay_ms, cfg.res_freq, cfg.res_gain_db]})
        print(f"[{i+1:3d}/{args.n}] bypass {m_by:+6.2f}  greedy {m_gr - m_by:+5.2f}  "
              f"oracle {m_or - m_by:+5.2f}  learned {m_lp - m_by:+5.2f} dB", flush=True)

        if i == 0:
            import soundfile as sf
            Path(args.audio_out).mkdir(parents=True, exist_ok=True)
            g = m_by + 8.0
            for name, supp in (("bypassed", None),
                               ("oracle", bank_supp(bank, hz, sr)),
                               ("learned", bank.as_suppressor(f, d))):
                mic, _ = loop(speech, g, suppressor=supp)
                sf.write(f"{args.audio_out}/{name}.wav",
                         mic[0].detach().cpu().numpy(), sr)
            export_preset(f[0], d[0], sr, "faust/learned_preset.dsp")

    add = lambda k: np.array([r[k] - r["bypass"] for r in rows])
    summary = {k: {"mean_added_db": float(add(k).mean()),
                   "std_added_db": float(add(k).std())}
               for k in ("greedy", "oracle", "learned")}
    print("\n" + "=" * 58)
    for k, v in summary.items():
        print(f"  {k:8s} added stable gain: {v['mean_added_db']:+6.2f} "
              f"+/- {v['std_added_db']:.2f} dB")
    print("=" * 58)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({"summary": summary, "rows": rows}, indent=2))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
