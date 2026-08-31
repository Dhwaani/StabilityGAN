#!/usr/bin/env python3
"""train the notch policy against the frozen surrogate.

The MetricGAN+ pattern, with MSG headroom in place of PESQ:

    L_D = |D(x) - headroom_true|^2      (step 03, done)
    L_G = |D(suppressed) - 1.0|^2       (here)
      + speech-quality term, so the policy cannot buy stability by
        notching the signal into silence

Alternating updates: the discriminator is refreshed on the policy's own
outputs each epoch (MetricGAN+ "historical set" trick) so it does not go
stale as the generator moves.

    python scripts/04_train_policy.py --data data/train.pt \
        --stabilitynet checkpoints/stabilitynet.pt
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np, torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from stabilitygan.data import load_speech_files, speech_batch
from stabilitygan.models import NotchPolicy, StabilityNet
from stabilitygan.msg import measure_msg
from stabilitygan.notch import NotchBank
from stabilitygan.plant import CabinConfig, CabinPath, ClosedLoop

HEADROOM_MIN, HEADROOM_MAX = -20.0, 20.0
def to_unit(db): return ((db - HEADROOM_MIN) / (HEADROOM_MAX - HEADROOM_MIN)).clamp(0, 1)


def speech_preservation(mic_sup, speech):
    """Penalise destroying the wanted signal. Log-spectral distance."""
    w = torch.hann_window(512, device=mic_sup.device)
    a = torch.stft(mic_sup, 512, 128, window=w, return_complex=True).abs().clamp_min(1e-6)
    b = torch.stft(speech, 512, 128, window=w, return_complex=True).abs().clamp_min(1e-6)
    return (a.log() - b.log()).pow(2).mean()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stabilitynet", default="checkpoints/stabilitynet.pt")
    ap.add_argument("--speech-dir", default=None)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--slots", type=int, default=4)
    ap.add_argument("--filter-width", type=int, default=31)
    ap.add_argument("--notch-width", type=float, default=0.08)
    ap.add_argument("--seconds", type=float, default=1.0)
    ap.add_argument("--frame", type=int, default=256)
    ap.add_argument("--sample-rate", type=int, default=16000)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lambda-quality", type=float, default=0.1)
    ap.add_argument("--headroom-target", type=float, default=8.0,
                    help="dB above bypassed MSG the policy is asked to hold")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="checkpoints/policy.pt")
    args = ap.parse_args()

    dev = torch.device(args.device)
    rng = np.random.default_rng(args.seed); torch.manual_seed(args.seed)
    sr = args.sample_rate
    n = int(args.seconds * sr); n -= n % args.frame

    disc = StabilityNet().to(dev)
    ck = torch.load(args.stabilitynet, weights_only=False, map_location=dev)
    disc.load_state_dict(ck["state_dict"])
    print(f"loaded surrogate (val MAE {ck.get('val_mae_db', float('nan')):.2f} dB)")

    policy = NotchPolicy(n_slots=args.slots).to(dev)
    opt_g = torch.optim.Adam(policy.parameters(), lr=args.lr)
    opt_d = torch.optim.Adam(disc.parameters(), lr=args.lr / 3)
    bank = NotchBank(args.slots, args.filter_width, args.notch_width).to(dev)
    files = load_speech_files(args.speech_dir)

    for step in range(1, args.steps + 1):
        cfg = CabinConfig.random(rng, sr)
        loop = ClosedLoop(CabinPath(cfg), frame_size=args.frame).to(dev)
        speech = speech_batch(args.batch, n, sr, files, rng).to(dev)

        with torch.no_grad():
            msg_bypass = float(measure_msg(loop, speech[:1], None, sr))
        gain = msg_bypass + args.headroom_target

        # ---- generator ------------------------------------------------
        probe, _ = loop(speech, gain, suppressor=None)
        freq, depth = policy(probe.detach())
        mic_sup, _ = loop(speech, gain, suppressor=bank.as_suppressor(freq, depth))

        d_hat = disc(mic_sup)
        loss_g = (d_hat - 1.0).pow(2).mean() \
            + args.lambda_quality * speech_preservation(mic_sup, speech)
        opt_g.zero_grad(); loss_g.backward(); opt_g.step()

        # ---- discriminator refresh on the policy's own output ---------
        if step % 5 == 0:
            with torch.no_grad():
                f2, d2 = policy(probe)
                sup2, _ = loop(speech[:1], gain, suppressor=bank.as_suppressor(f2[:1], d2[:1]))
                true_msg = float(measure_msg(loop, speech[:1], bank.as_suppressor(f2[:1], d2[:1]), sr))
            tgt = to_unit(torch.tensor([true_msg - gain], device=dev))
            loss_d = (disc(sup2) - tgt).pow(2).mean()
            opt_d.zero_grad(); loss_d.backward(); opt_d.step()

        if step % 50 == 0:
            hz = (freq[0] * sr / 2).detach().cpu().numpy()
            print(f"step {step:5d}  L_G {float(loss_g):7.4f}  D {float(d_hat.mean()):.3f}  "
                  f"notches {np.round(hz).tolist()} depths "
                  f"{np.round(depth[0].detach().cpu().numpy(), 2).tolist()}", flush=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": policy.state_dict(), "slots": args.slots,
                "filter_width": args.filter_width, "notch_width": args.notch_width,
                "sample_rate": sr}, args.out)
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
