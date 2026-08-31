PY ?= python
SPEECH ?=

.PHONY: help verify dataset surrogate policy eval demo test clean

help:
	@echo "make verify     - Day 1 gate: classical baselines"
	@echo "make dataset    - Day 2: generate MSG-labelled data"
	@echo "make surrogate  - Day 3 gate: train StabilityNet"
	@echo "make policy     - Day 4: train the notch policy"
	@echo "make eval       - Day 5: held-out results + audio"
	@echo "make demo       - Gradio app"
	@echo "make test       - unit tests"
	@echo ""
	@echo "pass a speech corpus with:  make dataset SPEECH=~/data/LibriSpeech"

verify:
	$(PY) scripts/01_verify_plant.py $(if $(SPEECH),--speech-dir "$(SPEECH)",)

dataset:
	$(PY) scripts/02_make_dataset.py --n 2000 --out data/train.pt --seed 0 $(if $(SPEECH),--speech-dir "$(SPEECH)",)
	$(PY) scripts/02_make_dataset.py --n 300  --out data/val.pt   --seed 99 $(if $(SPEECH),--speech-dir "$(SPEECH)",)

surrogate:
	$(PY) scripts/03_train_stabilitynet.py --train data/train.pt --val data/val.pt

policy:
	$(PY) scripts/04_train_policy.py $(if $(SPEECH),--speech-dir "$(SPEECH)",)

eval:
	$(PY) scripts/05_evaluate.py --n 30 $(if $(SPEECH),--speech-dir "$(SPEECH)",)

demo:
	$(PY) scripts/app.py

test:
	$(PY) -m pytest tests -q

clean:
	rm -rf data checkpoints audio docs/results.json docs/baselines.json faust/learned_preset.dsp
