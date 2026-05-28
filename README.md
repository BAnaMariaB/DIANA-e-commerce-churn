# DIANA — E-Commerce Customer Churn Prediction

**EPITA · AI Project Methodology 2025-2026**

A production-ready machine-learning system that predicts which e-commerce customers are at risk of leaving the platform 30 days before they actually churn, so Marketing and Customer Success can intervene in time. The system is fully trained, served, explainable, and deployed live on Azure.

> **Live API:** <https://diana-churn-api.greenfield-eca9baa0.northeurope.azurecontainerapps.io>
> **Swagger UI:** <https://diana-churn-api.greenfield-eca9baa0.northeurope.azurecontainerapps.io/docs>

---

## 1. Why this project

### The business problem

E-commerce platforms compete on retention, not acquisition. Industry studies put the cost of acquiring a new customer at 5–7× the cost of retaining an existing one, and loyal customers are 4–7× more profitable to serve over their lifetime. Even a 5 percentage-point reduction in churn typically translates to a 25–95 % increase in profit (Reichheld, *The Loyalty Effect*).

Yet most companies still react to churn — they only realize a customer is gone when the customer stops showing up. By then it's too late and far more expensive to win them back.

### The opportunity for AI

Churn is, by its nature, a **leading-indicator** problem. The signs are usually there in the customer's behavior before they churn: a drop in app usage, a recent complaint, a slowdown in order frequency, a delivery problem. A classification model can pick up these signals weeks before a human analyst would.

**DIANA** ("Data, Insights, Analytics & Architecture") is our implementation of this idea: an end-to-end ML system that combines transactional, behavioral, and demographic signals to score each customer's churn risk, attribute that risk to interpretable drivers, and feed the result back into retention workflows.

### Why the case for AI is specifically strong here

- The signals are quantifiable and already collected (orders, sessions, complaints).
- The cost of a wrong prediction is bounded: a false positive sends a marketing email; a false negative is a lost customer.
- The output is actionable at the individual level (per-customer probability), not just at the segment level.
- Modern explainability tools (SHAP) let us tell Customer Success teams *why* a customer is at risk, not just *that* they are.

---

## 2. What the application does

DIANA is composed of five layers, each independently runnable:

| # | Layer | What it does |
|---|---|---|
| 1 | **Data preparation** | Reads the raw Kaggle e-commerce dataset (5 630 customers × 20 features), imputes 822 missing values, normalizes inconsistent category labels, drops the non-predictive `CustomerID`. |
| 2 | **Feature engineering** | Bins continuous variables (`NumberOfAddress`, `CouponUsed`, `OrderCount`, `OrderAmountHikeFromlastYear`), one-hot encodes categoricals. Output: 37 features ready for the model. |
| 3 | **Model training & registry** | Trains three baselines (Logistic Regression, Random Forest, XGBoost), logs everything to **MLflow** with parameters, metrics, model artifacts. Registers the three models in the **MLflow Model Registry** with version tracking. |
| 4 | **Serving** | Production FastAPI inference server with `/health`, `/info`, `/predict` endpoints. Ships as a multi-stage Docker image. Deployed to **Azure Container Apps** with scale-to-zero — €0 when idle, auto-scales to 2 replicas under load. |
| 5 | **Explainability** | SHAP TreeExplainer over the trained XGBoost model produces waterfall, force, beeswarm, mean-\|SHAP\|, summary-per-class, and dependence plots for every customer or the full dataset. Satisfies GDPR right-to-explanation. |

Everything is wrapped in a Makefile so the whole thing runs with `make e2e`, and a pytest suite (22 tests) verifies each layer in isolation.

---

## 3. Findings

### 3.1 Model performance

Three models were trained on the same 80/20 stratified train/test split (4 504 train / 1 126 test rows). All numbers are on the held-out test set.

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression (baseline) | 88.7 % | 72.3 % | 53.7 % | 0.616 | 0.885 |
| Random Forest | 97.8 % | 99.4 % | 87.4 % | 0.930 | 0.999 |
| **XGBoost** (production) | **99.0 %** | **98.4 %** | **95.8 %** | **0.971** | **0.9995** |

**Selected model: XGBoost.** It dominates on every metric, particularly recall — meaning we catch 96 % of actual churners. Precision @ top-decile is also high, so the retention budget is spent on customers who will actually churn.

### 3.2 What drives churn (SHAP global drivers)

Computed via `shap.TreeExplainer` over the full test set (see `reports/figures/shap/beeswarm.png` and `mean_shap_bar.png`). Top five features ranked by mean |SHAP| value:

| Rank | Feature | How it relates to churn |
|---|---|---|
| 1 | **Tenure** | Low-tenure customers (< 1 year) are dramatically more likely to churn. The single strongest signal. |
| 2 | **Complain** | A recent complaint flag triples the average risk. Captures dissatisfaction not visible in transactions. |
| 3 | **NumberOfAddress** | Customers with many shipping addresses tend to churn more — possible proxy for indecision, gift-buyers, or address fraud. |
| 4 | **CashbackAmount** | Low cashback correlates with low engagement. Customers who don't earn cashback also don't come back. |
| 5 | **WarehouseToHome** | Longer delivery distances → more dissatisfaction → more churn. Confirms logistics is a retention lever, not just a cost center. |

These five features alone explain ~70 % of the model's total prediction variance.

### 3.3 What this means for the business

Translating the model's findings into a retention playbook:

- **Onboarding investment matters.** Tenure dominates; customers in their first year are fragile. Suggests doubling down on early-life CRM touchpoints (welcome series, second-purchase nudges).
- **Service recovery is high-leverage.** Every complaint that isn't resolved well becomes a churner. A complaint-resolution SLA + proactive follow-up is justified by the SHAP numbers alone.
- **Logistics is a retention investment, not just an OpEx line.** Customers far from a warehouse churn more. Opening a closer fulfillment center has measurable retention upside, not only delivery-cost upside.
- **Cashback is sticky.** Customers who use cashback come back. Worth segmenting and protecting in any cost-cutting cycle.

### 3.4 What we'd do next (productionization roadmap)

- **Time-aware split for production.** The current model uses random split; the production model should use a temporal split (train on T-3 months → validate T-2 → test T-1) to surface concept drift early.
- **Calibration check.** XGBoost probabilities tend to be miscalibrated near the extremes; a Platt-scaling or isotonic-regression layer would help downstream retention-cost decisions.
- **Fairness audit.** No fairness metrics were computed in this project. Before any production rollout, performance parity should be measured across `Gender`, `CityTier`, and age-band proxies.
- **A/B test the retention actions, not just the predictions.** The model is necessary but not sufficient; the actual ROI is measured by the lift in retention from the actions we take on its output.

---

## 4. Architecture

```
                   ┌──────────────┐
                   │ Kaggle CSV   │  (raw)
                   └──────┬───────┘
                          ▼
      ┌────────────────────────────────────────┐
      │ src/data_preparation/prepare.py        │   822 imputations
      │ src/feature_engineering/features.py    │   5 630 × 37 matrix
      └────────────────────┬───────────────────┘
                           ▼
                ┌──────────────────────┐
                │ src/training/        │
                │   train.py (MLflow)  ├──▶ mlruns/  +  Model Registry
                │   save_artifact.py   ├──▶ models/xgboost_model.json
                └──────────┬───────────┘
                           │
            ┌──────────────┴───────────────────┐
            ▼                                  ▼
   ┌─────────────────┐               ┌────────────────────┐
   │ src/inference/  │               │ src/explainability │
   │  predict.py     │               │  explain.py (SHAP) │
   │  FastAPI        │               │  15 plots          │
   └────────┬────────┘               └────────────────────┘
            ▼
   ┌─────────────────┐    docker build   ┌──────────────────┐
   │   Dockerfile    │ ─────────────────▶│ Azure Container  │
   │  multi-stage    │   docker push     │ Registry (ACR)   │
   └─────────────────┘                   └────────┬─────────┘
                                                  │ pulls image
                                                  ▼
                                        ┌──────────────────────┐
                                        │ Azure Container Apps │
                                        │ scale-to-zero, HTTPS │
                                        └──────────────────────┘
```

---

## 5. Project Structure

```
DIANA-e-commerce-churn/
├── data/
│   ├── raw/                        # Kaggle Excel (not versioned)
│   ├── processed/                  # Cleaned and encoded CSVs (not versioned)
│   └── external/                   # Reserved for external enrichment
├── notebooks/                      # EDA and exploration
├── src/
│   ├── data_preparation/
│   │   ├── prepare.py              # Cleaning + imputation
│   │   └── statistical_analysis.py # t-test, chi-square, Mann-Whitney
│   ├── feature_engineering/
│   │   └── features.py             # Binning + one-hot encoding
│   ├── training/
│   │   ├── train.py                # 3-model training + MLflow tracking
│   │   └── save_artifact.py        # Portable bundle for Docker
│   ├── inference/
│   │   └── predict.py              # FastAPI server (/health /info /predict)
│   └── explainability/
│       └── explain.py              # SHAP TreeExplainer + 15 plots
├── models/                         # Trained artifacts (xgboost_model.json, …)
├── reports/
│   ├── figures/shap/               # SHAP plots (committed for grader)
│   └── screenshots/                # Screenshots used in the report
├── azure/
│   ├── deploy.sh                   # One-shot Azure CLI deployment
│   ├── teardown.sh                 # Tear down the resource group
│   ├── sample_request.json         # Example /predict body
│   ├── .env.example                # Template for azure/.env
│   └── README.md                   # Cloud architecture + cost notes
├── .github/workflows/
│   └── deploy-azure.yml            # CI/CD → ACR → Container Apps
├── tests/                          # 22 pytest cases
├── scripts/run_e2e.sh              # End-to-end runner
├── docs/                           # Sphinx documentation
├── Dockerfile                      # Multi-stage build (train → serve)
├── MLproject                       # MLflow Project entry points
├── Makefile                        # `make test`, `make e2e`, `make api`, …
├── README.md / TESTING.md          # This file + testing walkthrough
├── pyproject.toml / poetry.lock    # Poetry dependency management
└── .flake8                         # PEP8 line length config
```

---

## 6. How to run the whole application

### 6.1 Prerequisites

- **macOS or Linux** (tested on macOS 14, Apple Silicon)
- **Python 3.12** (Poetry will pick this up; the project rejects 3.13+)
- **Poetry ≥ 2.4** — install with `curl -sSL https://install.python-poetry.org | python3 -`
- **Mac only:** `libomp` for XGBoost — `brew install libomp`
- **For the live deployment only:** Docker Desktop, Azure CLI, an Azure subscription

### 6.2 First-time setup

```bash
git clone <repo-url>
cd DIANA-e-commerce-churn

poetry lock
poetry install --extras dev
```

**Download the dataset** from [Kaggle](https://www.kaggle.com/datasets/ankitverma2010/ecommerce-customer-churn-analysis-and-prediction) and place the Excel file at:

```
data/raw/ECommerceDataset2.xlsx
```

(The brief says raw data isn't versioned; the `.gitignore` excludes it.)

### 6.3 Run everything — one command

```bash
make e2e
```

This runs the whole pipeline against the real Kaggle data and ends with `🎉 End-to-end pipeline passed.` after ~90 seconds. It will:

1. Validate the prerequisites
2. Run data preparation → `data/processed/churn_cleaned.csv`
3. Run feature engineering → `data/processed/churn_features.csv`
4. Train the XGBoost model + export the portable artifact bundle to `models/`
5. Run SHAP and write 15 plots to `reports/figures/shap/`
6. Run the pytest suite (22 tests)
7. Smoke-test the FastAPI server in-process

### 6.4 Run components individually

```bash
# Tests only (no real data needed — uses synthetic fixtures, ~4 s)
make test

# Just the training pipeline (with MLflow tracking)
poetry run python src/training/train.py

# Just the SHAP analysis
poetry run python -m src.explainability.explain

# Start the FastAPI server on http://localhost:8000
make api
```

Then in another terminal:

```bash
curl -s http://localhost:8000/health
# → {"status":"ok"}

curl -s -X POST http://localhost:8000/predict \
     -H 'Content-Type: application/json' \
     --data @azure/sample_request.json | python3 -m json.tool
```

Open <http://localhost:8000/docs> in your browser for the interactive Swagger UI.

### 6.5 MLflow experiments and Model Registry

After running `poetry run python src/training/train.py` at least once:

```bash
poetry run mlflow ui
# Opens MLflow on http://127.0.0.1:5000
```

In the UI:
- **Model training tab → Experiments → `churn-prediction`** — full run history, metrics, parameters.
- **Models** — the three registered models (XGBoost, Random Forest, Logistic Regression) with version history.

### 6.6 Reproducible runs with MLflow Projects

```bash
poetry run mlflow run . -e train         --env-manager=local
poetry run mlflow run . -e save_artifact --env-manager=local
poetry run mlflow run . -e explain       --env-manager=local
poetry run mlflow run . -e full_pipeline --env-manager=local
```

### 6.7 Local Docker container

Same image that's deployed on Azure. Trains the model during build so the served weights match what was tested:

```bash
make docker-build           # ~5–10 min; trains inside the image
make docker-run             # serves on http://localhost:8000
```

### 6.8 Live deployment on Azure (cloud bonus)

The model already runs in production at:

**<https://diana-churn-api.greenfield-eca9baa0.northeurope.azurecontainerapps.io>**

To re-deploy or deploy your own instance:

```bash
cp azure/.env.example azure/.env
# Edit azure/.env (subscription ID + unique ACR name)
bash azure/deploy.sh
```

Full architecture, cost notes (~€0/day on scale-to-zero), CI/CD setup, and troubleshooting in [`azure/README.md`](azure/README.md).

### 6.9 Local serving via MLflow's built-in server (alternative path)

```bash
poetry run mlflow models serve -m "models:/XGBoost/3" --port 5001 --no-conda
```

This satisfies the brief's "Deploy your model for serving on a local inference server" requirement using MLflow's native server rather than FastAPI. Either works.

### 6.10 View the Sphinx documentation

```bash
poetry run sphinx-build docs/source docs/build
open docs/build/index.html
```

### 6.11 Code quality

```bash
poetry run black src/ tests/    # auto-format
poetry run flake8 src/ tests/   # PEP8 lint
```

### 6.12 Testing walkthrough

A full checkpoint-by-checkpoint guide (local → Docker → Azure verification) lives in [`TESTING.md`](TESTING.md).

---

## 7. Tools & Best Practices Map

Each requirement in the assignment brief, mapped to its implementation:

| Brief requirement | Implementation |
|---|---|
| GIT for code & models versioning | Git + GitHub |
| Cookiecutter-style structure | `src/` with separated `data_preparation/`, `feature_engineering/`, `training/`, `inference/`, `explainability/` |
| PEP8 | Black + Flake8 + Pylint, configured in `.flake8` and `pyproject.toml` |
| Dependency management | Poetry (`pyproject.toml` + `poetry.lock`) |
| Sphinx documentation | `docs/source/` configured for `apidoc` |
| MLflow tracking | `src/training/train.py` logs params, metrics, models; `mlruns/` |
| MLflow Projects | `MLproject` file with 5 entry points |
| MLflow Model Registry | Three registered models (XGBoost, RF, LR) with versioned releases |
| MLflow local serving | `mlflow models serve -m "models:/XGBoost/3"` |
| **Cloud deployment (bonus)** | **Azure Container Apps, scale-to-zero, HTTPS, live URL above** |
| SHAP TreeExplainer | `src/explainability/explain.py` |
| SHAP plots (waterfall, force × 2, summary × 2, beeswarm, mean, dependence) | 15 files in `reports/figures/shap/` |
| Pytest | 22 tests in `tests/`, run with `make test` |
| CI/CD (extra) | `.github/workflows/deploy-azure.yml` |

---

## 8. Dataset

[E-Commerce Customer Churn Analysis & Prediction](https://www.kaggle.com/datasets/ankitverma2010/ecommerce-customer-churn-analysis-and-prediction) on Kaggle (Ankit Verma).

- **Rows:** 5 630 customers
- **Features:** 20 raw → 37 after one-hot encoding
- **Target:** `Churn` (binary; ≈ 17 % positive class)
- **Notable variables:** Tenure, PreferredLoginDevice, CityTier, WarehouseToHome, HourSpendOnApp, OrderCount, CashbackAmount, SatisfactionScore, Complain, …

The full data dictionary is documented inline in `src/data_preparation/prepare.py` and `src/feature_engineering/features.py`.

---

## 9. Authors

- **Ana-Maria Borduselu**
- **Kruthi Shandilya**
- **Ran Navlani**

EPITA — International Programs · AI Project Methodology 2025-2026
