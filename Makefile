# Convenience aliases for macOS / Linux. Every target delegates to run.py, which also works on
# Windows:  python run.py <task>
.PHONY: setup fetch-model test smoke dashboard api evaluate signals modules deck train data label clean

PYTHON ?= python3.11
VENV = .venv
PY = $(VENV)/bin/python

setup:
	$(PYTHON) -m venv $(VENV)
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install -r requirements.txt

fetch-model test smoke dashboard api evaluate signals modules deck train data label:
	$(PY) run.py $@

clean:
	rm -rf .pytest_cache .cache dist
	find . -name __pycache__ -type d -prune -not -path "./.venv/*" -exec rm -rf {} +
