"""Speech sources and dataset assembly.

No dataset agreement is required anywhere in this project. Speech comes
from LibriSpeech or VCTK (both CC BY 4.0); cabins are generated, not
measured; and MSG labels are produced by our own sweep. If no speech
folder is supplied the code falls back to a synthetic speech-like signal
so the repository runs end to end out of the box.
"""

from __future__ import annotations

import glob
import os
from typing import Optional

import numpy as np
import torch
from scipy import signal as sps


def synth_speech(n_samples: int, sample_rate: int, rng: np.random.Generator) -> np.ndarray:
    """A speech-like test signal: formant-filtered noise with a syllable envelope.

    Not a substitute for real speech in the final numbers, but enough to
    exercise the loop, and it makes the repo runnable with zero downloads.
    """
    x = rng.standard_normal(n_samples)
    for f0, q, g in ((500, 6, 1.0), (1500, 8, 0.6), (2600, 10, 0.3)):
        b, a = sps.iirpeak(min(f0, 0.45 * sample_rate) / (sample_rate / 2), q)
        x = x + g * sps.lfilter(b, a, x)

    # syllable-rate amplitude envelope, ~4 Hz
    t = np.arange(n_samples) / sample_rate
    env = 0.5 * (1.0 + np.sin(2 * np.pi * 4.0 * t + rng.uniform(0, 6.28)))
    env = np.clip(env, 0.05, None) ** 1.5
    x = x * env

    peak = np.abs(x).max()
    return (0.3 * x / peak).astype(np.float32) if peak > 0 else x.astype(np.float32)


def load_speech_files(folder: Optional[str]) -> list[str]:
    if not folder:
        return []
    pats = ("**/*.wav", "**/*.flac")
    files: list[str] = []
    for p in pats:
        files += glob.glob(os.path.join(folder, p), recursive=True)
    return sorted(files)


def read_clip(path: str, n_samples: int, sample_rate: int, rng: np.random.Generator) -> np.ndarray:
    import soundfile as sf

    info = sf.info(path)
    if info.frames <= n_samples:
        x, sr = sf.read(path, dtype="float32", always_2d=False)
    else:
        start = int(rng.integers(0, info.frames - n_samples))
        x, sr = sf.read(path, start=start, frames=n_samples, dtype="float32", always_2d=False)
    if x.ndim > 1:
        x = x.mean(axis=1)
    if sr != sample_rate:
        x = sps.resample_poly(x, sample_rate, sr).astype(np.float32)
    if len(x) < n_samples:
        x = np.pad(x, (0, n_samples - len(x)))
    x = x[:n_samples]
    peak = np.abs(x).max()
    return (0.3 * x / peak).astype(np.float32) if peak > 0 else x


def speech_batch(
    batch: int,
    n_samples: int,
    sample_rate: int,
    files: list[str],
    rng: np.random.Generator,
) -> torch.Tensor:
    out = []
    for _ in range(batch):
        if files:
            out.append(read_clip(str(rng.choice(files)), n_samples, sample_rate, rng))
        else:
            out.append(synth_speech(n_samples, sample_rate, rng))
    return torch.from_numpy(np.stack(out))
