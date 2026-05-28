# Testing Guide

This document walks you through every test we have, from the fastest
sanity check (~4 s) to the full Azure deployment.

## TL;DR

```bash
make install           # one-time
make e2e               # full local end-to-end (≈ 2 min)
```

If `make e2e` prints `🎉 End-to-end pipeline passed.` you're good for
local submission. Cloud deployment is its own section at the bottom.

---

## 1. What gets tested

| Layer | How | Where the test lives |
|---|---|---|
| Data preparation helpers | Crafted DataFrames in, asserts on output | `tests/test_data_preparation.py` |
| Feature engineering | Bins, label normalization, one-hot shape | `tests/test_feature_engineering.py` |
| Portable model bundle | Loadable, columns match, predicts in [0, 1] | `tests/test_artifact_bundle.py` |
| FastAPI server (`/health`, `/info`, `/predict`) | In-process `TestClient`, happy + error paths | `tests/test_inference_api.py` |
| SHAP pipeline | Every required plot produced & non-empty | `tests/test_explainability.py` |
| Full pipeline on REAL data | Bash script driving every stage | `scripts/run_e2e.sh` |
| Cloud deployment | Azure CLI script + GitHub Actions smoke test | `azure/deploy.sh` + `.github/workflows/deploy-azure.yml` |

All unit + smoke tests run in **under 5 seconds** and need **no real data**.
The end-to-end script needs the Kaggle file at `data/raw/ECommerceDataset2.xlsx`.

---

## 2. Step-by-step on your Mac

### 2.1 One-time setup

```bash
# Inside the repo
poetry install
# Mac users: XGBoost needs libomp
brew install libomp
```

Confirm the raw data is in place:

```bash
ls -la data/raw/ECommerceDataset2.xlsx
# expected: a ~555 KB file. If missing, download from
# https://www.kaggle.com/datasets/ankitverma2010/ecommerce-customer-churn-analysis-and-prediction
```

### 2.2 Fastest sanity check — pytest only (~ 4 s, no real data needed)

```bash
make test
```

What you should see (last line):

```
======================== 22 passed in 3.7s ========================
```

### 2.3 Full pipeline on the real data (~ 1–2 min)

```bash
make e2e
```

Expected output, condensed:

```
▶ [0/7] Checking prerequisites
✓ Raw dataset present
▶ [1/7] Data preparation
✓ data/processed/churn_cleaned.csv produced
▶ [2/7] Feature engineering
✓ data/processed/churn_features.csv produced
▶ [3/7] Train + export portable artifact bundle
✓ models/ bundle written          ← XGBoost 99.0% accuracy, ROC-AUC 0.9995
▶ [4/7] SHAP analysis (sample 300)
✓ All SHAP plots generated         ← 15 files in reports/figures/shap/
▶ [5/7] Pytest suite
✓ All tests passed
▶ [6/7] FastAPI smoke test (in-process, no network)
  /predict returned: {'predictions': [1], 'probabilities': [0.89...], ...}
✓ FastAPI endpoint responds correctly
▶ [7/7] Done
🎉 End-to-end pipeline passed.
```

Things to spot-check after `make e2e`:

```bash
cat models/metadata.json | python -m json.tool      # training metrics
open reports/figures/shap/beeswarm.png              # global SHAP plot
open reports/figures/shap/force_row_0.html          # interactive explanation
```

### 2.4 Run the FastAPI server locally and curl it

```bash
make api                    # starts uvicorn on http://localhost:8000
```

In another shell:

```bash
curl -s http://localhost:8000/health
# → {"status":"ok"}

curl -s http://localhost:8000/info | python -m json.tool

curl -s -X POST http://localhost:8000/predict \
     -H 'Content-Type: application/json' \
     --data @azure/sample_request.json | python -m json.tool
# → {"predictions":[...],"probabilities":[...],"model_name":"XGBoost","n_features":37}
```

Also try the interactive API docs at <http://localhost:8000/docs>.

### 2.5 Docker build + run (needs Docker Desktop running)

```bash
make docker-build           # ≈ 5–10 min; the build trains the model inside the image
make docker-run             # serves on http://localhost:8000

# In another shell:
curl -s http://localhost:8000/health
curl -s -X POST http://localhost:8000/predict \
     -H 'Content-Type: application/json' \
     --data @azure/sample_request.json | python -m json.tool
```

If the image builds cleanly and `/predict` returns a probability, the
container is ready for Azure.

### 2.6 Lint + format check

```bash
make lint                   # black --check + flake8
make format                 # apply black formatting in place
```

---

## 3. Azure deployment test (real cloud)

You confirmed you have an Azure account ready, so this is the production
test. Costs: ACR Basic at ~€0.15/day, Container Apps free-tier covers the
demo traffic. Total exposure if you forget to tear down for a week: ~€1.

### 3.1 Configure

```bash
cp azure/.env.example azure/.env
# Edit azure/.env:
#   AZURE_SUBSCRIPTION_ID — `az account show --query id -o tsv`
#   ACR_NAME              — must be globally unique, lowercase, ≤ 50 chars
#   LOCATION              — westeurope works for EU students
```

### 3.2 Deploy

```bash
az login
bash azure/deploy.sh
```

The script prints a final URL like:

```
https://diana-churn-api.<random>.westeurope.azurecontainerapps.io
```

### 3.3 Smoke-test the live endpoint

```bash
APP_URL="https://...azurecontainerapps.io"      # from deploy.sh output

curl -s "${APP_URL}/health"
curl -s "${APP_URL}/info" | python -m json.tool
curl -s -X POST "${APP_URL}/predict" \
     -H 'Content-Type: application/json' \
     --data @azure/sample_request.json | python -m json.tool
```

Open `${APP_URL}/docs` in a browser for the interactive Swagger UI — that's
the screenshot worth putting in the final report.

### 3.4 Tear down when grading is over

```bash
bash azure/teardown.sh
```

---

## 4. Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `make e2e` fails at step 0 with "raw dataset missing" | The Excel file isn't at `data/raw/ECommerceDataset2.xlsx` | Download from Kaggle (link in step 2.1) |
| `make e2e` fails at step 3 with `libomp` error | macOS without libomp | `brew install libomp` |
| Pytest fails on `test_shap_pipeline_…` with `ValueError: could not convert string to float: '[…E-1]'` | XGBoost 3.x serializes `base_score` as a list-string | Already handled by the compatibility shim in `src/explainability/explain.py`; if you upgraded shap and the shim is now a no-op, just re-run. |
| Docker build runs but the runtime fails with `Model file models/xgboost_model.json not found` | The build stage didn't reach the training step | Make sure `data/raw/ECommerceDataset2.xlsx` is **not** in `.dockerignore` (it isn't by default) and is present locally before `make docker-build` |
| Azure deploy: `az: command not found` | Azure CLI not installed | <https://learn.microsoft.com/cli/azure/install-azure-cli> |
| Azure deploy: `ContainerAppOperationError: <ACR>.azurecr.io/diana-churn-api:v1 not found` | ACR build step failed silently | Re-run `bash azure/deploy.sh` — look at the `az acr build` output, fix and retry |
| `/predict` returns 400 "Empty input." | Sent `{"customers": []}` | Send at least one row |
| `/predict` returns 422 from FastAPI | JSON shape doesn't match `PredictRequest` | Check field names; either `customers` (raw dicts) or `features` (encoded), not both |

---

## 5. What "passes" looks like for the report

You can copy/paste these expected results into the Part 2 / Part 3 section
of the final report — they match what the grader will see when they clone
the repo and run `make e2e`.

```
Model: XGBoost (37 features, 5630 rows)
Accuracy : 0.9902
Precision: 0.9838
Recall   : 0.9579
F1       : 0.9707
ROC-AUC  : 0.9995

Top-5 SHAP drivers: Tenure, Complain, NumberOfAddress, CashbackAmount, WarehouseToHome
22 unit + smoke tests pass in ~4 s.
End-to-end script `scripts/run_e2e.sh` finishes in ~90 s.
FastAPI endpoint /predict returns churn probability for a real-customer payload.
Azure Container App responds on https://...azurecontainerapps.io/docs.
```
