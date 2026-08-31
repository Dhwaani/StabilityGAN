"""Maximum stable gain (MSG) measurement.

This is the metric the whole project is built around, and the reason it
needs a learned surrogate: **MSG is not differentiable**. It is obtained by
sweeping loop gain until the loop howls, which is a bifurcation, not a
smooth function you can backpropagate through.

MetricGAN solved exactly this shape of problem for PESQ. StabilityNet
(``models.StabilityNet``) is the same idea applied to MSG.

Howling test
------------
Because ``loudspeaker_sat`` limits the loop, an unstable system does not
diverge to infinity — it settles into a limit cycle. So "did it blow up"
is not a reliable test. Instead:

1. Drive the loop with speech.
2. Append one second of **silence**.
3. Measure the RMS of the tail.

A stable loop decays into the silence. An unstable one sustains a tone.
This is both crisp and physically what "howling" means.
"""

from __future__ import annotations

import torch

from .plant import ClosedLoop

SILENCE_SEC = 1.0
HOWL_THRESHOLD_DB = -60.0


def tail_level_db(
    loop: ClosedLoop,
    speech: torch.Tensor,
    loop_gain_db: float,
    suppressor=None,
    sample_rate: int = 16000,
) -> torch.Tensor:
    """Run the loop with a silent tail; return tail RMS in dBFS. Shape (B,)."""
    B = speech.shape[0]
    n_sil = int(SILENCE_SEC * sample_rate)
    n_sil -= n_sil % loop.frame_size
    padded = torch.cat(
        [speech, torch.zeros(B, n_sil, device=speech.device, dtype=speech.dtype)], dim=-1
    )
    mic, _ = loop(padded, loop_gain_db, suppressor=suppressor)
    tail = mic[:, -n_sil:]
    rms = tail.pow(2).mean(dim=-1).clamp_min(1e-20).sqrt()
    return 20.0 * torch.log10(rms)


@torch.no_grad()
def measure_msg(
    loop: ClosedLoop,
    speech: torch.Tensor,
    suppressor=None,
    sample_rate: int = 16000,
    lo_db: float = -20.0,
    hi_db: float = 40.0,
    coarse_step: float = 2.0,
    refine: bool = True,
) -> torch.Tensor:
    """Sweep loop gain and return MSG in dB. Shape (B,).

    Coarse linear sweep, then a bisection refine to 0.25 dB. This is the
    Python port of ``dsp/icc_msg_probe.dsp``.
    """
    B = speech.shape[0]
    device = speech.device
    msg = torch.full((B,), lo_db, device=device)
    found = torch.zeros(B, dtype=torch.bool, device=device)

    g = lo_db
    while g <= hi_db:
        lvl = tail_level_db(loop, speech, g, suppressor, sample_rate)
        howling = lvl > HOWL_THRESHOLD_DB
        newly = howling & (~found)
        msg = torch.where(newly, torch.full_like(msg, g - coarse_step), msg)
        found = found | howling
        if bool(found.all()):
            break
        g += coarse_step

    msg = torch.where(found, msg, torch.full_like(msg, hi_db))

    if refine:
        for _ in range(3):
            probe = msg + coarse_step / 2.0
            lvl = tail_level_db(loop, speech, float(probe.mean()), suppressor, sample_rate)
            ok = lvl <= HOWL_THRESHOLD_DB
            msg = torch.where(ok, probe, msg)
            coarse_step /= 2.0

    return msg


def added_stable_gain(msg_suppressed: torch.Tensor, msg_bypassed: torch.Tensor) -> torch.Tensor:
    """The headline number: how many dB of stable gain the suppressor bought.

    faust-icc measured +4 dB bypassed, +16 dB suppressed = 12 dB added, on
    the default cabin at 16 kHz.
    """
    return msg_suppressed - msg_bypassed


@torch.no_grad()
def probe_howl_frequencies(
    loop: ClosedLoop,
    speech: torch.Tensor,
    gain_db: float,
    n_peaks: int,
    sample_rate: int = 16000,
    suppressor=None,
    min_sep_hz: float = 80.0,
    prominence_db: float = 6.0,
):
    """Find where the loop actually rings, in Hz.

    Drive the loop above MSG, then FFT the silent tail. The peaks are the
    modes that survive, which is what ``ic.howlDetect`` looks for -- and
    crucially they are *not* the same as the peaks of the cabin magnitude
    response, because the loop delay imposes a comb and the saturation
    generates odd harmonics.

    Returns a numpy array of frequencies, ascending, possibly shorter than
    ``n_peaks``.
    """
    import numpy as np
    from scipy import signal as sps

    n_sil = int(SILENCE_SEC * sample_rate)
    n_sil -= n_sil % loop.frame_size
    padded = torch.cat(
        [speech, torch.zeros(speech.shape[0], n_sil, device=speech.device, dtype=speech.dtype)],
        dim=-1,
    )
    mic, _ = loop(padded, gain_db, suppressor=suppressor)
    tail = mic[0, -n_sil:].detach().cpu().numpy()

    spec = np.abs(np.fft.rfft(tail * np.hanning(len(tail))))
    freqs = np.fft.rfftfreq(len(tail), 1.0 / sample_rate)
    bin_hz = freqs[1] if len(freqs) > 1 else 1.0
    peaks, _ = sps.find_peaks(
        20 * np.log10(spec + 1e-12),
        distance=max(1, int(min_sep_hz / bin_hz)),
        prominence=prominence_db,
    )
    if len(peaks) == 0:
        return np.array([])
    top = peaks[np.argsort(spec[peaks])[::-1]][:n_peaks]
    return np.sort(freqs[top])
