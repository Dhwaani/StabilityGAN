# StabilityGAN

**Differentiable howling suppression, trained against a learned maximum-stable-gain surrogate.**

An extension of [faust-icc](https://github.com/Dhwaani/In-CarCommunication).

[![python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org)
[![license](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![no dataset agreement](https://img.shields.io/badge/datasets-no%20agreement%20required-green.svg)](#data)

---

## The idea 

MetricGAN showed that you can optimize non-differentiable metrics like PESQ by training a discriminator to predict the metric and optimizing the speech enhancer against that predictor.

Acoustic feedback suppression faces the exact same challenge with Maximum Stable Gain (MSG). MSG is measured by ramping up loop gain until the system breaks into sustained howling—a non-differentiable bifurcation point. 
As a result, systems like `faust-icc` rely on hand-tuned heuristic thresholds to allocate notch filters. StabilityGAN adapts the MetricGAN framework to acoustic stability: it replaces quality prediction with a learned MSG surrogate network, allowing backpropagation to directly guide optimal notch placement.

## Status

| | |
|---|---|
| Plant, notch bank, MSG sweep | working, unit-tested |
| Classical baselines | measured — see [docs/validation.md](docs/validation.md) |
| StabilityNet + policy training | implemented, **not yet trained to a result** |

The bar the learned policy has to clear is **+7.75 dB** of added stable gain, which is what oracle notch placement achieves on the default cabin. The greedy self-allocating cascade — the behaviour of `ic.notchBank` in `faust-icc` —
reaches +6.00 dB.

---

## Setup

Tested on Python 3.10–3.12, Linux and macOS.

```bash
git clone <your-fork-url> stabilitygan
cd stabilitygan

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install --upgrade pip
pip install -r requirements.txt
```

**CPU-only torch** (smaller download, fine for steps 1 and 5):

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

Verify:

```bash
python -c "import torch, speechbrain; print(torch.__version__, torch.cuda.is_available())"
make test
```

Expect `6 passed`.

### Speech data (optional)

Everything runs without any download — there is a synthetic speech-like fallback.
For real numbers, point the scripts at a corpus. Neither needs an agreement:

```bash
# LibriSpeech dev-clean, ~340 MB, CC BY 4.0
wget https://www.openslr.org/resources/12/dev-clean.tar.gz
tar xzf dev-clean.tar.gz

export SPEECH=$PWD/LibriSpeech/dev-clean
```

---

## Run it

```bash
make help
```

### Establish the baselines (~2 min, CPU)

```bash
make verify                       # or: make verify SPEECH=$SPEECH
```

```
[1] bypassed MSG           :  +0.00 dB
    howl modes (Hz)        : [914.3, 2741.9, 4569.6, 6398.2]
[2] oracle notch bank MSG  :  +7.75 dB   (added +7.75 dB)
[3] greedy cascade MSG     :  +6.00 dB   (added +6.00 dB)
GATE PASSED
```

**This is a gate.** If the numbers are not close to these, stop and fix the plant
before generating data. Everything downstream is referenced to them.

### Build the labelled dataset (hours, run on the GPU box)

```bash
make dataset SPEECH=$SPEECH
```

Each label costs a full gain sweep, so this is the expensive step. Run it once
and cache. Reduce `--n` to smoke-test:

```bash
python scripts/02_make_dataset.py --n 20 --out data/smoke.pt
```

### Train the surrogate (~30 min on a GPU)

```bash
make surrogate
```

**This is a gate.** Validation MAE must come in under 3 dB. If it does not, the
GAN in step 4 cannot work, and you have found that out while training the surrogate.

### Train the policy

```bash
make policy SPEECH=$SPEECH
```

### The result

```bash
make eval SPEECH=$SPEECH
```

Writes `docs/results.json`, three WAV renders into `audio/`, and a FAUST preset
into `faust/learned_preset.dsp`.

### The demo

```bash
make demo                          # http://127.0.0.1:7860
python scripts/app.py --share      
```

Upload speech, push the loop past its stability limit, hear the howl, hear it
suppressed, read the measured numbers.

---

## What connects to what

| SpeechBrain | Used for |
|---|---|
| `processing.signal_processing.notch_filter` | the notch construction, adapted to accept tensors so gradients reach the placement |
| `processing.signal_processing.reverberate` | the pattern for the cabin convolution |
| `recipes/Voicebank/enhance/MetricGAN` | the discriminator-as-surrogate training scheme |

| faust-icc | Ported to |
|---|---|
| `ic.cabinPath` | `plant.CabinPath` |
| `ic.loudspeakerSat` | `plant.loudspeaker_sat` |
| `ic.closedLoop` | `plant.ClosedLoop` |
| `ic.howlDetect` | `msg.probe_howl_frequencies` (baseline) → `models.NotchPolicy` (learned) |
| `ic.notchBank` | `notch.NotchBank` |
| `dsp/icc_msg_probe.dsp` | `msg.measure_msg` |

`faust_export.py` writes the learned notches back out as a FAUST preset.
**The neural policy is a design tool; `faust-icc` stays the real-time
deployment target.**

## The finding worth knowing before you start

A notch bank adds its own group delay *inside the loop it is stabilising*, which
moves the howling frequency off the notch just placed there. Measured, with
oracle placement:

| filter_width | added stable gain |
|---|---|
| 31 | **+7.75 dB** |
| 101 | +5.50 dB |
| 513 | **−8.00 dB** |

Long sharp notches are worse than doing nothing. This is why the defaults are
short, and it is the specific weakness of a greedy cascade that a learned policy
might exploit. Details in [docs/method.md](docs/method.md).

## Layout

```
stabilitygan/
├── stabilitygan/
│   ├── plant.py          torch port of the faust-icc acoustic loop
│   ├── notch.py          differentiable notch bank (adapted from SpeechBrain)
│   ├── msg.py            MSG sweep + howl-mode probe (the non-differentiable metric)
│   ├── models.py         StabilityNet (surrogate) + NotchPolicy (generator)
│   ├── data.py           speech loading, synthetic fallback, cabin sampling
│   └── faust_export.py   learned notches → FAUST preset
├── scripts/01..05, app.py
├── tests/                6 unit tests
└── docs/                 method.md, validation.md
```

## Citing

```bibtex
@software{stabilitygan,
  author = {Chakraborty, Ashmita},
  title  = {StabilityGAN: Differentiable Howling Suppression via a Learned
            Maximum-Stable-Gain Surrogate},
  year   = {2026},
  url    = {https://github.com/Dhwaani/StabilityGAN}
}
```

Builds directly on:

- Fu, Yu, Hsieh, Plantinga, Ravanelli, Lu, Tsao. *MetricGAN+: An Improved
  Version of MetricGAN for Speech Enhancement.* Interspeech 2021.
- Ravanelli et al. *SpeechBrain: A General-Purpose Speech Toolkit.* 2021.
- van Waterschoot & Moonen. *Fifty Years of Acoustic Feedback Control: State of
  the Art and Future Challenges.* Proc. IEEE 99(2), 2011.

## License

MIT — see [LICENSE](LICENSE). SpeechBrain is Apache-2.0; the adapted `notch_filter` construction is credited in `notch.py`.
