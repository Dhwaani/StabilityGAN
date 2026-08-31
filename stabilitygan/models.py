"""StabilityNet (discriminator) and NotchPolicy (generator).

Both follow the MetricGAN+ pattern
(Fu et al., Interspeech 2021; recipe in SpeechBrain at
``recipes/Voicebank/enhance/MetricGAN``):

* the **discriminator** learns a differentiable surrogate for a metric that
  is otherwise non-differentiable — PESQ there, MSG here;
* the **generator** is then trained to drive the surrogate towards 1.0.

MSG is normalised to [0, 1] over ``MSG_MIN..MSG_MAX`` dB so the losses are
scaled the same way MetricGAN+ scales PESQ.
"""

from __future__ import annotations

import torch
import torch.nn as nn

MSG_MIN_DB = -20.0
MSG_MAX_DB = 40.0


def msg_to_unit(msg_db: torch.Tensor) -> torch.Tensor:
    return ((msg_db - MSG_MIN_DB) / (MSG_MAX_DB - MSG_MIN_DB)).clamp(0.0, 1.0)


def unit_to_msg(u: torch.Tensor) -> torch.Tensor:
    return u * (MSG_MAX_DB - MSG_MIN_DB) + MSG_MIN_DB


class LogSTFT(nn.Module):
    """Log-magnitude STFT front end shared by both networks."""

    def __init__(self, n_fft: int = 512, hop: int = 128):
        super().__init__()
        self.n_fft, self.hop = n_fft, hop
        self.register_buffer("window", torch.hann_window(n_fft))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        spec = torch.stft(
            x, self.n_fft, self.hop, window=self.window,
            return_complex=True, center=True,
        )
        return torch.log1p(spec.abs())              # (B, F, T)


class StabilityNet(nn.Module):
    """Predicts maximum stable gain from a loop recording.

    This is the differentiable stand-in for ``msg.measure_msg``. It sees a
    short recording of the loop running and outputs normalised MSG.
    """

    def __init__(self, n_fft: int = 512, hop: int = 128):
        super().__init__()
        self.front = LogSTFT(n_fft, hop)
        ch = [1, 16, 32, 64, 64]
        blocks = []
        for i in range(len(ch) - 1):
            blocks += [
                nn.Conv2d(ch[i], ch[i + 1], 3, stride=2, padding=1),
                nn.InstanceNorm2d(ch[i + 1]),
                nn.PReLU(),
            ]
        self.conv = nn.Sequential(*blocks)
        self.head = nn.Sequential(
            nn.Linear(ch[-1], 64), nn.PReLU(), nn.Linear(64, 1), nn.Sigmoid()
        )

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        h = self.front(wav).unsqueeze(1)             # (B,1,F,T)
        h = self.conv(h)
        h = h.mean(dim=(2, 3))                       # global average pool
        return self.head(h).squeeze(-1)              # (B,)


class NotchPolicy(nn.Module):
    """Emits K notch frequencies and depths for a loop recording.

    Replaces the hand-tuned detector + self-allocating cascade in
    ``ic.howlDetect`` / ``ic.notchBank`` with a learned joint allocation.
    """

    def __init__(self, n_slots: int = 4, n_fft: int = 512, hop: int = 128):
        super().__init__()
        self.n_slots = n_slots
        self.front = LogSTFT(n_fft, hop)
        n_freq = n_fft // 2 + 1
        self.enc = nn.Sequential(
            nn.Conv1d(n_freq, 128, 5, padding=2), nn.PReLU(),
            nn.Conv1d(128, 128, 5, padding=2), nn.PReLU(),
        )
        self.gru = nn.GRU(128, 128, batch_first=True, bidirectional=False)
        self.freq_head = nn.Sequential(nn.Linear(128, n_slots), nn.Sigmoid())
        self.depth_head = nn.Sequential(nn.Linear(128, n_slots), nn.Sigmoid())

    def forward(self, wav: torch.Tensor):
        """Return (freq, depth), each (B, K).

        ``freq`` is a fraction of Nyquist in (0, 1); ``depth`` is in [0, 1].
        A single decision per clip keeps the loop cheap and matches how the
        FAUST slots latch onto a mode rather than tracking it sample by
        sample.
        """
        h = self.front(wav)                          # (B,F,T)
        h = self.enc(h).transpose(1, 2)              # (B,T,C)
        h, _ = self.gru(h)
        h = h[:, -1]                                 # last state
        freq = self.freq_head(h).clamp(0.02, 0.98)
        depth = self.depth_head(h)
        return freq, depth
