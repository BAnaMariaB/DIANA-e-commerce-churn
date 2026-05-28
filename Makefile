# DIANA — Makefile
# One-shot commands for the most common dev + test workflows.
#
# Quick reference:
#   make install      Poetry install with all extras
#   make pipeline     prep → features → train → save_artifact
#   make api          Run the FastAPI server locally on :8000
#   make explain      Generate every SHAP plot (reports/figures/shap/)
#   make test         Run the pytest suite (no real data needed)
#   make test-real    Run the pytest suite + the real-data E2E script
#   make docker-build Build the Dockerfile locally (needs Docker Desktop)
#   make docker-run   Run the local image on :8000
#   make e2e          End-to-end smoke test (calls scripts/run_e2e.sh)
#   make lint         Black + flake8 over src/ and tests/
#   make clean        Remove build / cache directories (keeps models + data)

POETRY ?= poetry
PY ?= python

.PHONY: help install pipeline prep features train save-artifact api explain \
        test test-real docker-build docker-run e2e lint format clean

help:
	@awk 'BEGIN { FS=":.*##" } /^[a-zA-Z_-]+:.*##/ { printf "  %-15s %s\n", $$1, $$2 }' $(MAKEFILE_LIST)

install: ## Install all Python dependencies via Poetry
	$(POETRY) install

# ── Pipeline ──
prep: ## Data preparation only
	$(POETRY) run $(PY) -m src.data_preparation.prepare

features: ## Feature engineering only
	$(POETRY) run $(PY) -m src.feature_engineering.features

train: ## MLflow training (writes to mlruns/)
	$(POETRY) run $(PY) -m src.training.train

save-artifact: ## Train + export the portable XGBoost bundle to models/
	$(POETRY) run $(PY) -m src.training.save_artifact

pipeline: prep features save-artifact ## Full local pipeline (no MLflow run wrapper)
	@echo "✓ Pipeline finished — models/xgboost_model.json is ready."

# ── Serving ──
api: ## Run the FastAPI inference server locally on http://localhost:8000
	$(POETRY) run uvicorn src.inference.predict:app --host 0.0.0.0 --port 8000 --reload

# ── XAI ──
explain: ## Generate every SHAP plot (reports/figures/shap/)
	$(POETRY) run $(PY) -m src.explainability.explain

# ── Tests ──
test: ## Run the unit + smoke test suite (no real data needed)
	$(POETRY) run pytest tests/ -v

test-real: pipeline ## Run pytest then the real-data E2E script
	$(POETRY) run pytest tests/ -v
	bash scripts/run_e2e.sh

e2e: ## End-to-end test (data → features → train → API → SHAP)
	bash scripts/run_e2e.sh

# ── Docker ──
docker-build: ## Build the Docker image locally
	docker build -t diana-churn-api:local .

docker-run: ## Run the local Docker image on :8000
	docker run --rm -p 8000:8000 -e PORT=8000 diana-churn-api:local

# ── Code quality ──
lint: ## Run black --check + flake8 on src/ and tests/
	$(POETRY) run black --check src/ tests/
	$(POETRY) run flake8 src/ tests/

format: ## Apply black to src/ and tests/
	$(POETRY) run black src/ tests/

# ── Cleanup ──
clean: ## Remove caches, mlruns, build artifacts (keeps models/ and data/)
	rm -rf .pytest_cache .mypy_cache .ruff_cache
	find . -type d -name __pycache__ -exec rm -rf {} +
	rm -rf mlruns/ docs/_build/ docs/build/
	@echo "✓ Cleaned."
