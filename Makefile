PY := .venv/bin/python

.PHONY: setup data eda bench test clean

setup:
	python3 -m venv .venv
	$(PY) -m pip install -r requirements.txt

data:
	$(PY) scripts/prepare_data.py

eda:
	$(PY) scripts/run_eda.py

bench:
	$(PY) scripts/run_benchmark.py

test:
	$(PY) -m pytest -q tests

clean:
	rm -rf results .pytest_cache
