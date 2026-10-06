.PHONY: setup data label train evaluate signals modules dashboard api deck test clean

PYTHON = python3.11
VENV = .venv
PY = $(VENV)/bin/python
OFFLINE = HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false TRANSFORMERS_VERBOSITY=error

setup:
	$(PYTHON) -m venv $(VENV)
	$(VENV)/bin/pip install --upgrade pip
	$(VENV)/bin/pip install -r requirements.txt

# Rebuild the labelled training table from data/raw (needs the Kaggle downloads)
data:
	$(PY) -m scripts.build_unified_dataset

# Zero-shot BART-MNLI event labels (slow: ~1 hour on Apple MPS), then merge into the dataset
label:
	$(PY) -m scripts.relabel_events_zeroshot

train:
	$(OFFLINE) $(PY) -m scripts.train_multitask

evaluate:
	$(OFFLINE) $(PY) -m scripts.evaluate_engine

signals:
	$(OFFLINE) $(PY) -m scripts.generate_signals

modules:
	$(PY) -m scripts.build_synthetic_portfolio
	$(PY) -m scripts.run_module_a
	$(PY) -m scripts.run_module_b

dashboard:
	$(OFFLINE) $(VENV)/bin/streamlit run app/dashboard.py

api:
	$(OFFLINE) $(VENV)/bin/uvicorn src.api.main:app --host 0.0.0.0 --port 8000

deck:
	$(PY) -m scripts.build_deck
	dot -Tpng docs/architecture.dot -o docs/architecture.png

test:
	$(VENV)/bin/pytest tests/ -q

clean:
	rm -rf __pycache__ .pytest_cache
