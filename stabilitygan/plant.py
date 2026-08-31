"""Differentiable closed-loop ICC plant.

This is a PyTorch port of the acoustic loop in ``faust-icc``
(https://github.com/Dhwaani/In-CarCommunication), specifically:

    ic.cabinPath      -> CabinPath
    ic.loudspeakerSat -> torch.tanh
    ic.closedLoop     -> ClosedLoop

The port must be numerically faithful, because every measurement in this
project is referenced to the maximum-stable-gain (MSG) numbers produced by
the FAUST implementation. ``scripts/01_verify_plant.py`` is the gate.

Design notes
------------
* The cabin path is designed as an IIR (delay + resonance + HF roll-off),
  then rendered to a finite impulse response and **normalised to 0 dB peak
  magnitude**. That normalisation is what makes a loop-gain reading in dB
  mean something physical: |L| = 1 sits at 0 dB, so the Nyquist condition
  is at the number you read off the dial.
* Everything downstream is a convolution or a pointwise nonlinearity, so
  the whole loop is differentiable end to end.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from scipy import signal as sps


@dataclass
class CabinConfig:
    """Parameters of the synthetic cabin transfer function.

    Mirrors ``ic.cabinPath(delayMs, resFreq, resGainDb, hfCutHz)``.
    """

    sample_rate: int = 16000
    delay_ms: float = 8.0
    res_freq: float = 900.0
    res_gain_db: float = 12.0
    res_q: float = 8.0
    hf_cut_hz: float = 4000.0
    ir_len: int = 1024

    @staticmethod
    def random(rng: np.random.Generator, sample_rate: int = 16000) -> "CabinConfig":
        """Sample a plausible cabin. Used to build a multi-room dataset."""
        return CabinConfig(
            sample_rate=sample_rate,
            delay_ms=float(rng.uniform(3.0, 20.0)),
            res_freq=float(rng.uniform(300.0, 2500.0)),
            res_gain_db=float(rng.uniform(6.0, 18.0)),
            res_q=float(rng.uniform(4.0, 16.0)),
            hf_cut_hz=float(rng.uniform(2500.0, 6000.0)),
        )


def design_cabin_ir(cfg: CabinConfig) -> np.ndarray:
    """Render the cabin transfer function to a peak-normalised FIR.

    Returns
    -------
    np.ndarray
        Impulse response of length ``cfg.ir_len``, scaled so that
        ``max |H(f)| == 1.0`` (0 dB).
    """
    fs = cfg.sample_rate
    nyq = fs / 2.0

    # --- resonance: a peaking biquad -------------------------------------
    w0 = 2.0 * math.pi * min(cfg.res_freq, 0.95 * nyq) / fs
    amp = 10.0 ** (cfg.res_gain_db / 40.0)
    alpha = math.sin(w0) / (2.0 * cfg.res_q)
    b = np.array([1 + alpha * amp, -2 * math.cos(w0), 1 - alpha * amp])
    a = np.array([1 + alpha / amp, -2 * math.cos(w0), 1 - alpha / amp])
    b, a = b / a[0], a / a[0]

    # --- HF roll-off: one-pole lowpass -----------------------------------
    x = math.exp(-2.0 * math.pi * min(cfg.hf_cut_hz, 0.95 * nyq) / fs)
    b_lp = np.array([1.0 - x])
    a_lp = np.array([1.0, -x])

    # --- render to an impulse response -----------------------------------
    delay = int(round(cfg.delay_ms * 1e-3 * fs))
    imp = np.zeros(cfg.ir_len, dtype=np.float64)
    if delay >= cfg.ir_len:
        raise ValueError("delay_ms exceeds ir_len; increase CabinConfig.ir_len")
    imp[delay] = 1.0

    ir = sps.lfilter(b, a, imp)
    ir = sps.lfilter(b_lp, a_lp, ir)

    # --- normalise to 0 dB peak magnitude --------------------------------
    # This is the bug fix recorded in faust-icc docs/tuning.md: without it,
    # the cabin has passband gain of its own and loop-gain dB readings stop
    # being physically interpretable.
    mag = np.abs(np.fft.rfft(ir, n=8192))
    peak = float(mag.max())
    if peak <= 0.0:
        raise RuntimeError("degenerate cabin impulse response")
    ir = ir / peak

    return ir.astype(np.float32)


class CabinPath(torch.nn.Module):
    """Fixed FIR cabin path with streaming state.

    Equivalent to SpeechBrain's ``reverberate`` but stateful, so it can be
    driven frame by frame inside a closed loop.
    """

    def __init__(self, cfg: CabinConfig):
        super().__init__()
        self.cfg = cfg
        ir = design_cabin_ir(cfg)
        self.register_buffer("ir", torch.from_numpy(ir).flip(0).view(1, 1, -1))
        self.taps = self.ir.shape[-1]

    def init_state(self, batch: int, device=None, dtype=torch.float32):
        return torch.zeros(batch, self.taps - 1, device=device, dtype=dtype)

    def forward(self, frame: torch.Tensor, state: torch.Tensor):
        """Convolve one frame.

        Parameters
        ----------
        frame : (B, N)
        state : (B, taps-1) tail from the previous frame

        Returns
        -------
        out : (B, N)
        state : (B, taps-1)
        """
        x = torch.cat([state, frame], dim=-1).unsqueeze(1)  # (B,1,taps-1+N)
        out = F.conv1d(x, self.ir).squeeze(1)               # (B, N)
        new_state = torch.cat([state, frame], dim=-1)[:, -(self.taps - 1):]
        return out, new_state


def loudspeaker_sat(x: torch.Tensor, drive: float = 1.0) -> torch.Tensor:
    """Soft saturation, so an unstable loop rings instead of reaching NaN.

    Port of ``ic.loudspeakerSat``. Without this the MSG sweep produces inf
    and the gradient is undefined everywhere past the stability boundary.
    """
    return torch.tanh(drive * x) / drive


class ClosedLoop(torch.nn.Module):
    """One ICC direction wrapped in an acoustic loop.

    Signal path per frame::

        mic  = speech + feedback
        y    = suppressor(mic)
        ls   = sat(loop_gain * y)
        feedback[n+1] = cabin(ls)

    ``suppressor`` is any callable ``(frame, state) -> (frame, state)``. Pass
    ``None`` for the bypassed (no suppression) reference case.
    """

    def __init__(self, cabin: CabinPath, frame_size: int = 256):
        super().__init__()
        self.cabin = cabin
        self.frame_size = frame_size

    def forward(
        self,
        speech: torch.Tensor,
        loop_gain_db: torch.Tensor | float,
        suppressor=None,
        suppressor_state=None,
    ):
        """Run the loop.

        Parameters
        ----------
        speech : (B, T) dry near-end speech
        loop_gain_db : scalar or (B,) loop gain in dB
        suppressor : callable or None

        Returns
        -------
        mic : (B, T) microphone signal (what a listener at the far end hears)
        ls  : (B, T) loudspeaker signal
        """
        B, T = speech.shape
        device, dtype = speech.device, speech.dtype
        N = self.frame_size

        if not torch.is_tensor(loop_gain_db):
            loop_gain_db = torch.tensor(float(loop_gain_db), device=device, dtype=dtype)
        gain = (10.0 ** (loop_gain_db / 20.0)).reshape(-1, 1)
        if gain.shape[0] == 1:
            gain = gain.expand(B, 1)

        n_frames = T // N
        speech = speech[:, : n_frames * N]

        cab_state = self.cabin.init_state(B, device=device, dtype=dtype)
        feedback = torch.zeros(B, N, device=device, dtype=dtype)

        mic_out, ls_out = [], []
        for i in range(n_frames):
            frame = speech[:, i * N:(i + 1) * N]
            mic = frame + feedback

            if suppressor is None:
                y = mic
            else:
                y, suppressor_state = suppressor(mic, suppressor_state)

            ls = loudspeaker_sat(gain * y)
            feedback, cab_state = self.cabin(ls, cab_state)

            mic_out.append(mic)
            ls_out.append(ls)

        return torch.cat(mic_out, dim=-1), torch.cat(ls_out, dim=-1)
