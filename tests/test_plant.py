"""Unit tests. Run with: python -m pytest tests -q"""
import numpy as np
import torch

from stabilitygan.msg import measure_msg, probe_howl_frequencies
from stabilitygan.notch import NotchBank, notch_kernels
from stabilitygan.plant import CabinConfig, CabinPath, ClosedLoop, design_cabin_ir


def test_cabin_is_peak_normalised():
    ir = design_cabin_ir(CabinConfig())
    mag = np.abs(np.fft.rfft(ir, n=8192))
    assert abs(mag.max() - 1.0) < 1e-4, "cabin must be 0 dB peak or loop-gain dB is meaningless"


def test_notch_attenuates_at_centre():
    f = torch.tensor([[0.25]])
    k = notch_kernels(f, torch.ones(1, 1))
    H = np.abs(np.fft.rfft(k[0, 0].numpy(), n=4096))
    centre = int(0.25 * (len(H) - 1))
    assert H[centre] < 0.5, "notch does not attenuate its own centre frequency"


def test_depth_zero_is_passthrough():
    f = torch.tensor([[0.25]])
    k = notch_kernels(f, torch.zeros(1, 1))
    pad = k.shape[-1] // 2
    assert abs(float(k[0, 0, pad]) - 1.0) < 1e-5


def test_notch_gradients_flow_to_frequency():
    f = torch.tensor([[0.25]], requires_grad=True)
    d = torch.tensor([[1.0]], requires_grad=True)
    notch_kernels(f, d).sum().backward()
    assert f.grad is not None and torch.isfinite(f.grad).all()
    assert float(f.grad.abs()) > 0, "no gradient w.r.t. notch frequency"


def test_loop_is_finite_past_instability():
    loop = ClosedLoop(CabinPath(CabinConfig()), frame_size=256)
    x = torch.randn(1, 4096) * 0.1
    mic, _ = loop(x, 40.0)
    assert torch.isfinite(mic).all(), "saturation missing: loop reached inf/NaN"


def test_suppression_increases_msg():
    sr = 16000
    loop = ClosedLoop(CabinPath(CabinConfig(sample_rate=sr)), frame_size=256)
    rng = np.random.default_rng(0)
    x = torch.from_numpy(rng.standard_normal((1, sr - sr % 256)).astype(np.float32) * 0.1)
    m0 = float(measure_msg(loop, x, None, sr))
    hz = probe_howl_frequencies(loop, x, m0 + 6.0, 4, sr)
    assert len(hz) > 0
    hz = np.pad(hz, (0, 4 - len(hz)), mode="edge")
    bank = NotchBank(4)
    f = torch.tensor(hz.astype(np.float32) / (sr / 2)).clamp(0.02, 0.98).unsqueeze(0)
    m1 = float(measure_msg(loop, x, bank.as_suppressor(f, torch.ones(1, 4)), sr))
    assert m1 > m0, "notching the measured howl modes did not raise MSG"
