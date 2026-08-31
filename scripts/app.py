#!/usr/bin/env python3
"""Gradio demo -- upload speech, hear it howl, hear it suppressed.

    python scripts/app.py                  # local, http://127.0.0.1:7860
    python scripts/app.py --share          # temporary public link
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np, torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from stabilitygan.data import synth_speech
from stabilitygan.models import NotchPolicy
from stabilitygan.msg import measure_msg, probe_howl_frequencies
from stabilitygan.notch import NotchBank
from stabilitygan.plant import CabinConfig, CabinPath, ClosedLoop

SR, FRAME = 16000, 256


def build(policy_path):
    ck = torch.load(policy_path, weights_only=False, map_location="cpu")
    p = NotchPolicy(n_slots=ck["slots"]); p.load_state_dict(ck["state_dict"]); p.eval()
    return p, NotchBank(ck["slots"], ck["filter_width"], ck["notch_width"]), ck


def run(audio, extra_gain_db, delay_ms, res_freq, res_gain_db, policy, bank, ck):
    if audio is None:
        x = synth_speech(SR, SR, np.random.default_rng(0))
    else:
        sr_in, x = audio
        x = x.astype(np.float32)
        if x.ndim > 1:
            x = x.mean(axis=1)
        x = x / (np.abs(x).max() + 1e-9) * 0.3
        if sr_in != SR:
            from scipy import signal as sps
            x = sps.resample_poly(x, SR, sr_in).astype(np.float32)
    n = len(x) - (len(x) % FRAME)
    speech = torch.from_numpy(x[:n]).unsqueeze(0)

    cfg = CabinConfig(sample_rate=SR, delay_ms=delay_ms,
                      res_freq=res_freq, res_gain_db=res_gain_db)
    loop = ClosedLoop(CabinPath(cfg), frame_size=FRAME)

    m_by = float(measure_msg(loop, speech, None, SR))
    g = m_by + float(extra_gain_db)

    mic_by, _ = loop(speech, g, suppressor=None)
    with torch.no_grad():
        f, d = policy(mic_by)
    mic_sup, _ = loop(speech, g, suppressor=bank.as_suppressor(f, d))
    m_lp = float(measure_msg(loop, speech, bank.as_suppressor(f, d), SR))

    hz = np.round((f[0] * SR / 2).numpy(), 1).tolist()
    dep = np.round(d[0].numpy(), 2).tolist()
    report = (
        f"bypassed MSG      : {m_by:+.2f} dB\n"
        f"loop running at   : {g:+.2f} dB  ({extra_gain_db:+.1f} dB over MSG)\n"
        f"with policy MSG   : {m_lp:+.2f} dB\n"
        f"added stable gain : {m_lp - m_by:+.2f} dB\n\n"
        f"notches (Hz)      : {hz}\n"
        f"depths            : {dep}"
    )
    to_np = lambda t: (SR, t[0].detach().numpy())
    return to_np(mic_by), to_np(mic_sup), report


def main():
    import gradio as gr
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", default="checkpoints/policy.pt")
    ap.add_argument("--share", action="store_true")
    args = ap.parse_args()
    policy, bank, ck = build(args.policy)

    with gr.Blocks(title="StabilityGAN") as demo:
        gr.Markdown(
            "# StabilityGAN\n"
            "Differentiable howling suppression trained against a learned "
            "maximum-stable-gain surrogate. Extends "
            "[faust-icc](https://github.com/Dhwaani/In-CarCommunication).\n\n"
            "Push the loop past its stability limit and listen to what the "
            "learned notch policy does with it."
        )
        with gr.Row():
            with gr.Column():
                au = gr.Audio(label="speech (leave empty for a synthetic clip)")
                eg = gr.Slider(0, 16, value=8, step=0.5, label="gain over MSG (dB)")
                dm = gr.Slider(3, 20, value=8, step=0.5, label="cabin delay (ms)")
                rf = gr.Slider(300, 2500, value=900, step=10, label="cabin resonance (Hz)")
                rg = gr.Slider(6, 18, value=12, step=0.5, label="resonance gain (dB)")
                btn = gr.Button("run", variant="primary")
            with gr.Column():
                o1 = gr.Audio(label="bypassed -- this is the howl")
                o2 = gr.Audio(label="suppressed")
                txt = gr.Textbox(label="measurement", lines=9)
        btn.click(lambda a, e, d, r, g_: run(a, e, d, r, g_, policy, bank, ck),
                  [au, eg, dm, rf, rg], [o1, o2, txt])
    demo.launch(share=args.share)


if __name__ == "__main__":
    main()
