# Palm Guard — top-level developer entrypoints.
# Each target shells into the relevant package. See CLAUDE.md for the contract.

PY ?= python3
ML  := packages/ml
API := packages/api
EDGE := packages/edge
WEB := packages/web

.PHONY: help install data baseline train eval export api web test test-ml test-api lint clean

help:
	@echo "Palm Guard targets:"
	@echo "  make install    install Python deps for ml + api"
	@echo "  make data       generate synthetic RPW audio + manifest"
	@echo "  make baseline   train/eval classical RandomForest baseline"
	@echo "  make train      train transfer-learning CNN (needs TensorFlow)"
	@echo "  make eval       threshold-aware metrics on held-out site split"
	@echo "  make export     quantize to TFLite + parity check"
	@echo "  make api        run FastAPI backend on :8000"
	@echo "  make web        run Next.js dashboard on :3000"
	@echo "  make test       run ml + api test suites"

install:
	pip install -r $(ML)/requirements.txt
	pip install -r $(API)/requirements.txt

data:
	cd $(ML) && $(PY) -m palmguard_ml.cli data

baseline:
	cd $(ML) && $(PY) -m palmguard_ml.cli baseline

train:
	cd $(ML) && $(PY) -m palmguard_ml.cli train

eval:
	cd $(ML) && $(PY) -m palmguard_ml.cli eval

export:
	cd $(ML) && $(PY) -m palmguard_ml.cli export

api:
	cd $(API) && $(PY) -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

web:
	cd $(WEB) && npm run dev

test: test-ml test-api

test-ml:
	cd $(ML) && $(PY) -m pytest -q

test-api:
	cd $(API) && $(PY) -m pytest -q

lint:
	ruff check $(ML) $(API) packages/edge || true

clean:
	rm -rf data/raw data/processed $(ML)/artifacts
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
