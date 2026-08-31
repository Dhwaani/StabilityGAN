"""Differentiable notch bank.

Adapted from ``speechbrain.processing.signal_processing.notch_filter``
(SpeechBrain, Apache-2.0). SpeechBrain's version takes a Python float for
the notch frequency and builds one fixed kernel. This version accepts a
**tensor** of frequencies and depths with shape ``(B, K)``, so gradients
flow back to the notch placement — which is the whole point of the project.

The construction is the same as SpeechBrain's: a band-reject kernel is the
sum of a low-pass below the notch and a high-pass above it.

Depth
-----
A slot with depth 0 is a pass-through, depth 1 is a full notch. The kernel
is interpolated against a unit impulse::

    k = (1 - d) * delta + d * notch

so a slot can fade in continuously instead of switching, matching the
behaviour of ``ic.notchSlot`` in faust-icc (which fades on a confidence
value rather than a boolean, to avoid clicks).
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def _sinc(x: torch.Tensor) -> torch.Tensor:
    """sin(x)/x with the removable singularity at 0 handled.

    The obvious one-liner ``torch.where(x==0, 1, sin(x)/x)`` is WRONG here.
    ``torch.where`` evaluates both branches, so ``sin(0)/0`` produces a NaN
    that is discarded in the forward pass but poisons the backward pass --
    every gradient w.r.t. notch frequency comes back NaN. The input has to
    be masked *before* the division.
    """
    near_zero = x.abs() < 1e-8
    x_safe = torch.where(near_zero, torch.ones_like(x), x)
    return torch.where(near_zero, torch.ones_like(x), torch.sin(x_safe) / x_safe)


def _lowpass(cutoff: torch.Tensor, n: torch.Tensor, window: torch.Tensor) -> torch.Tensor:
    """Windowed-sinc low-pass. ``cutoff`` in (0,1) as a fraction of Nyquist."""
    c = cutoff.unsqueeze(-1)                       # (B, K, 1)
    k = _sinc(math.pi * c * n) * c * window        # (B, K, L)
    return k / k.sum(dim=-1, keepdim=True).clamp_min(1e-8)


def notch_kernels(
    freq: torch.Tensor,
    depth: torch.Tensor,
    width: float = 0.08,
    filter_width: int = 31,
) -> torch.Tensor:
    """Build a batch of notch kernels.

    Parameters
    ----------
    freq : (B, K) notch centres, fraction of sample_rate/2, in (0, 1)
    depth : (B, K) in [0, 1]
    width : notch width, fraction of sample_rate/2
    filter_width : odd kernel length

    Returns
    -------
    (B, K, filter_width)
    """
    if filter_width % 2 == 0:
        raise ValueError("filter_width must be odd")

    device, dtype = freq.device, freq.dtype
    pad = filter_width // 2
    n = torch.arange(filter_width, device=device, dtype=dtype) - pad
    window = torch.blackman_window(filter_width, periodic=False, device=device, dtype=dtype)

    lo = (freq - width).clamp(1e-3, 0.999)
    hi = (freq + width).clamp(1e-3, 0.999)

    k_lo = _lowpass(lo, n, window)

    # high-pass by spectral inversion of a low-pass at `hi`
    k_hi = -_lowpass(hi, n, window)
    # spectral inversion, out-of-place so autograd stays happy
    impulse = torch.zeros_like(k_hi)
    impulse[..., pad] = 1.0
    k_hi = k_hi + impulse

    notch = k_lo + k_hi                            # (B, K, L)

    delta = torch.zeros_like(notch)
    delta[..., pad] = 1.0

    d = depth.clamp(0.0, 1.0).unsqueeze(-1)
    return (1.0 - d) * delta + d * notch


class NotchBank(torch.nn.Module):
    """Cascade of K differentiable notch slots, applied frame by frame.

    Unlike ``ic.notchBank`` in faust-icc — where slot n detects on the
    output of slot n-1 and therefore allocates itself — the frequencies
    here are supplied from outside (by the learned policy). The cascade
    structure is kept so the two are directly comparable.
    """

    def __init__(self, n_slots: int = 4, filter_width: int = 31, width: float = 0.08):
        """
        Defaults matter here and were chosen by measurement, not taste.

        A notch bank adds its own group delay (``filter_width/2`` samples)
        *inside* the loop it is stabilising. That shifts the loop phase and
        moves the howling frequency off the notch. Measured on the default
        cabin: ``filter_width=31`` gives +7.75 dB of added stable gain,
        ``101`` gives +5.5 dB, and ``513`` gives -8 dB. Short wins.

        This self-defeating delay is the central problem the learned policy
        exists to solve: it can place notches knowing what its own delay
        will do to the plant. A greedy cascade cannot.
        """
        super().__init__()
        self.n_slots = n_slots
        self.filter_width = filter_width
        self.width = width
        self.pad = filter_width // 2

    def init_state(self, batch: int, device=None, dtype=torch.float32):
        return torch.zeros(batch, self.n_slots, self.filter_width - 1, device=device, dtype=dtype)

    def forward(self, frame: torch.Tensor, state, freq: torch.Tensor, depth: torch.Tensor):
        """Apply the cascade to one frame.

        Parameters
        ----------
        frame : (B, N)
        state : (B, K, filter_width-1) per-slot tail, or None
        freq, depth : (B, K)
        """
        B, N = frame.shape
        device, dtype = frame.device, frame.dtype
        if state is None:
            state = self.init_state(B, device=device, dtype=dtype)

        kernels = notch_kernels(freq, depth, self.width, self.filter_width)
        new_state = []
        x = frame
        for k in range(self.n_slots):
            tail = state[:, k]                                  # (B, L-1)
            padded = torch.cat([tail, x], dim=-1).unsqueeze(1)   # (B,1,L-1+N)
            ker = kernels[:, k].flip(-1).unsqueeze(1)            # (B,1,L)
            # grouped conv gives each batch item its own kernel
            y = F.conv1d(
                padded.reshape(1, B, -1),
                ker.reshape(B, 1, -1),
                groups=B,
            ).reshape(B, -1)
            new_state.append(torch.cat([tail, x], dim=-1)[:, -(self.filter_width - 1):])
            x = y
        return x, torch.stack(new_state, dim=1)

    def as_suppressor(self, freq: torch.Tensor, depth: torch.Tensor):
        """Return a ``(frame, state) -> (frame, state)`` callable for ClosedLoop."""

        def _fn(frame, state):
            return self.forward(frame, state, freq, depth)

        return _fn
