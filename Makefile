# Laptop: `make env setup data`. Pod without conda: `make setup data RUN=python`.
ENV = flycraftax
RUN ?= conda run -n $(ENV) python

env:
	conda create -y -n $(ENV) python=3.11

setup:
	$(RUN) -m pip install -e ".[test]"

data:
	$(RUN) scripts/download_data.py

test:
	$(RUN) -m pytest

test-slow:
	$(RUN) -m pytest -m slow

bench:
	$(RUN) scripts/bench_brain.py

mn9:
	$(RUN) scripts/mn9_check.py

calibrate:
	$(RUN) scripts/calibrate.py

eval:
	$(RUN) scripts/eval_zero_shot.py

train:
	$(RUN) scripts/train_ppo.py

viewer:
	$(RUN) scripts/make_viewer.py

.PHONY: env setup data test test-slow bench mn9 calibrate eval train viewer
