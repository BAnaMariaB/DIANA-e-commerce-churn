#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# DIANA — End-to-end local smoke test.
#
# Runs the entire pipeline against the real Kaggle dataset and asserts that
# each stage produces the expected artifacts. Designed to be run by anyone
# (team member or grader) with:
#
#   1. Poetry installed.
#   2. data/raw/ECommerceDataset2.xlsx in place.
#
# Reports pass/fail clearly per step and exits non-zero on the first failure.
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

# ── Pretty output helpers ──
GREEN="\033[32m"; RED="\033[31m"; YELLOW="\033[33m"; BLUE="\033[34m"; RESET="\033[0m"
step() { echo -e "${BLUE}▶ $1${RESET}"; }
ok()   { echo -e "${GREEN}✓ $1${RESET}"; }
fail() { echo -e "${RED}✗ $1${RESET}"; exit 1; }
warn() { echo -e "${YELLOW}! $1${RESET}"; }

# ── Pick the runner ──
if command -v poetry >/dev/null 2>&1 && [ -f "pyproject.toml" ]; then
    RUN="poetry run"
else
    warn "Poetry not found — falling back to plain python3."
    RUN="python3 -m"
fi

# Allow callers to override; sane default for poetry users.
PY="${RUN} python"
if [[ "${RUN}" == "python3 -m" ]]; then
    PY="python3"
fi

# ── 0. Prereqs ──
step "[0/7] Checking prerequisites"
[[ -f "data/raw/ECommerceDataset2.xlsx" ]] || fail \
    "data/raw/ECommerceDataset2.xlsx is missing. Download from Kaggle and try again."
ok "Raw dataset present"

# ── 1. Data preparation ──
step "[1/7] Data preparation"
${PY} -m src.data_preparation.prepare
[[ -f "data/processed/churn_cleaned.csv" ]] || fail "churn_cleaned.csv was not produced"
ok "data/processed/churn_cleaned.csv produced"

# ── 2. Feature engineering ──
step "[2/7] Feature engineering"
${PY} -m src.feature_engineering.features
[[ -f "data/processed/churn_features.csv" ]] || fail "churn_features.csv was not produced"
ok "data/processed/churn_features.csv produced"

# ── 3. Train + export portable artifact bundle ──
step "[3/7] Train + export portable artifact bundle"
${PY} -m src.training.save_artifact
for f in xgboost_model.json feature_columns.json metadata.json; do
    [[ -s "models/${f}" ]] || fail "models/${f} is missing or empty"
done
ok "models/ bundle written"

# ── 4. SHAP analysis ──
step "[4/7] SHAP analysis (sample 300)"
${PY} -m src.explainability.explain --sample 300
REQUIRED_SHAP="waterfall_row_0.png force_row_0.html force_row_0.png \
force_all_points.html beeswarm.png mean_shap_bar.png \
summary_class_0.png summary_class_1.png shap_values.npy expected_value.npy"
for f in ${REQUIRED_SHAP}; do
    [[ -s "reports/figures/shap/${f}" ]] || fail "missing/empty SHAP output: ${f}"
done
# At least 5 dependence plots
DEP_COUNT=$(ls reports/figures/shap/dependence_*.png 2>/dev/null | wc -l)
(( DEP_COUNT >= 5 )) || fail "expected ≥5 dependence plots, got ${DEP_COUNT}"
ok "All SHAP plots generated"

# ── 5. Pytest ──
step "[5/7] Pytest suite"
${PY} -m pytest tests/ -q
ok "All tests passed"

# ── 6. FastAPI smoke test ──
step "[6/7] FastAPI smoke test (in-process, no network)"
${PY} - <<'PY'
import os, sys, json
sys.path.insert(0, ".")
os.environ["MODELS_DIR"] = "models"
from fastapi.testclient import TestClient
from src.inference.predict import app
c = TestClient(app)
assert c.get("/health").json() == {"status": "ok"}, "health failed"
info = c.get("/info").json()
assert info["model_name"] == "XGBoost", "info.model_name mismatch"
# real customer payload
payload = {"customers": [{
    "Tenure": 1, "PreferredLoginDevice": "Mobile", "CityTier": 3,
    "WarehouseToHome": 12, "PreferredPaymentMode": "Debit Card",
    "Gender": "Female", "HourSpendOnApp": 4, "NumberOfDeviceRegistered": 4,
    "PreferedOrderCat": "Mobile", "SatisfactionScore": 1,
    "MaritalStatus": "Single", "NumberOfAddress": 9, "Complain": 1,
    "OrderAmountHikeFromlastYear": 12, "CouponUsed": 0, "OrderCount": 1,
    "DaySinceLastOrder": 8, "CashbackAmount": 120.0,
    "AddressGroup": "7-10", "CouponGroup": "0", "OrderCountGroup": "1-2",
    "OrderAmountHikeGroup": "Low",
}]}
r = c.post("/predict", json=payload)
assert r.status_code == 200, r.text
print(f"  /predict returned: {r.json()}")
PY
ok "FastAPI endpoint responds correctly"

# ── 7. Summary ──
step "[7/7] Done"
echo
echo -e "${GREEN}🎉 End-to-end pipeline passed.${RESET}"
echo
echo "Artifacts you can inspect:"
echo "  data/processed/churn_features.csv      — model-ready dataset"
echo "  models/xgboost_model.json              — portable trained model"
echo "  models/metadata.json                   — training metrics (cat it!)"
echo "  reports/figures/shap/beeswarm.png      — top-line SHAP summary"
echo "  reports/figures/shap/force_row_0.html  — open in a browser"
echo
echo "Next steps:"
echo "  make api                 # serve the model locally on :8000"
echo "  make docker-build && make docker-run    # full container test"
echo "  bash azure/deploy.sh     # ship it to Azure Container Apps"
