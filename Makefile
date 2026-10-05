.PHONY: setup test api clean

PYTHON = python3.11
VENV = .venv

setup:
	$(PYTHON) -m venv $(VENV)
	$(VENV)/bin/pip install --upgrade pip
	$(VENV)/bin/pip install -r requirements.txt

test:
	$(VENV)/bin/pytest tests/ -v

api:
	$(VENV)/bin/uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload

clean:
	rm -rf __pycache__ .pytest_cache $(VENV)
