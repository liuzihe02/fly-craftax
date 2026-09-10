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

.PHONY: env setup data test test-slow bench mn9
