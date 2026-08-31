# Method

## The problem with the metric

`faust-icc` reports its result as **added stable gain**: the difference in
maximum stable gain (MSG) between the bypassed loop and the suppressed one.
On the default cabin at 16 kHz that was +4 dB bypassed, +16 dB suppressed,
so 12 dB added.

That number is obtained by sweeping loop gain upward until the system
howls. Howling onset is a bifurcation. There is no gradient. 
We cannot put MSG in a loss function.

So the notch bank in `faust-icc` is hand-tuned: a 15 dB prominence
threshold, a 200 ms persistence window, a cascade where slot *n* detects on
the output of slot *n-1*. Every one of those constants was chosen by a
human looking at a spectrum.

## Borrowing the fix from MetricGAN

MetricGAN (Fu et al., ICML 2019; MetricGAN+, Interspeech 2021; recipe in
SpeechBrain at `recipes/Voicebank/enhance/MetricGAN`) faced the same shape
of problem with PESQ, which is also non-differentiable. Their solution:

A discriminator is trained to *predict* the metric. It is differentiable
by construction, so the generator can be trained against it. MetricGAN+
reaches PESQ 3.15 on VoiceBank-DEMAND this way.

**StabilityGAN applies that to a stability metric instead of a quality
metric.** As far as we can find, MetricGAN has been extended to PESQ,
STOI, DNSMOS, SIIB, ESTOI and the composite measures — all quality or
intelligibility, never to MSG.

## Architecture

```
┌──────────── Differentiable Acoustic Plant ───────────┐
  speech ──►(+)──┤  NotchBank ──► ×gain ──► tanh ──► CabinPath  ├──┐
                 └──────────────────────────────────────────────┘  │
                  ▲                                                │
                  └──────────────── Feedback ──────────────────────┘
                         │
                         ├──► NotchPolicy  ──► (freq, depth)  [Generator]
                         └──► StabilityNet ──► predicted headroom [Surrogate]
```

* `plant.py` is a faithful torch port of `ic.cabinPath`, `ic.loudspeakerSat`
  and `ic.closedLoop`. The cabin is normalised to 0 dB peak magnitude, so
  |L| = 1 sits at 0 dB and a loop-gain reading in dB means something.
* `notch.py` adapts SpeechBrain's `notch_filter` to accept **tensors** of
  frequency and depth, so gradients reach the notch placement.
* `models.StabilityNet` predicts remaining gain headroom in dB.
* `models.NotchPolicy` emits K frequencies and K depths.

Losses, following MetricGAN+ exactly:

```
L_D = |D(x) − headroom_true|²
L_G = |D(suppressed) − 1.0|²  +  λ · log-spectral-distance(suppressed, dry)
```

The quality term is not optional. Without it the policy discovers that the cheapest
 way to make a loop stable is to notch the signal into silence.

## The finding that shapes the whole design

A notch bank adds its own group delay — `filter_width/2` samples — *inside
the loop it is stabilising*. That shifts the loop phase, which moves the
howling frequency off the notch that was just placed there.

Measured on the default cabin, oracle placement on the true howl modes:

| filter_width | notch width | added stable gain |
|---|---|---|
| 31  | 0.08 | **+7.75 dB** |
| 101 | 0.05 | +5.50 dB |
| 257 | 0.02 | +2.50 dB |
| 513 | 0.012 | **−8.00 dB** |

Long, sharp notches make the system *worse than doing nothing*. This is
the same effect FAUST users hit when adding a filter to a waveguide loop
detunes the string.

It is also the reason a learned policy might beat a greedy cascade: the
policy can place notches knowing what its own delay will do to the plant.
A cascade that re-detects after each slot cannot — it chases a target that
its own previous slot has already moved. Measured, the greedy cascade
allocates slot 2 at 884 Hz after slot 1 took 914 Hz, then wastes slots 3
and 4 at 38 Hz, and ends up **worse** than oracle placement.

## Falsification Criteria

If the trained policy simply replicates oracle peak-frequency matching, the system provides no functional advantage over standard peak detectors. The core thesis holds only if the network intentionally places notch frequencies at specific offsets relative to the raw howl modes to compensate for internal loop delay
