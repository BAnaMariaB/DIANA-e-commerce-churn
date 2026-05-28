"""Pytest fixtures shared across the test suite.

All fixtures use small synthetic data + a tiny XGBoost model so the suite
runs in seconds and doesn't depend on the Kaggle dataset being present.
The end-to-end test against the real data lives in a separately marked
``real_data`` test that the Makefile runs only when the file exists.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from xgboost import XGBClassifier


# Real feature column list produced by features.py on the Kaggle dataset.
# Hard-coded here so unit tests don't need to run the whole prep pipeline.
FEATURE_COLUMNS: list[str] = [
    "Tenure",
    "CityTier",
    "WarehouseToHome",
    "HourSpendOnApp",
    "NumberOfDeviceRegistered",
    "SatisfactionScore",
    "NumberOfAddress",
    "Complain",
    "OrderAmountHikeFromlastYear",
    "CouponUsed",
    "OrderCount",
    "DaySinceLastOrder",
    "CashbackAmount",
    "PreferredLoginDevice_Mobile",
    "PreferredPaymentMode_Credit Card",
    "PreferredPaymentMode_Debit Card",
    "PreferredPaymentMode_E wallet",
    "PreferredPaymentMode_UPI",
    "Gender_Male",
    "PreferedOrderCat_Grocery",
    "PreferedOrderCat_Laptop & Accessory",
    "PreferedOrderCat_Mobile",
    "PreferedOrderCat_Others",
    "MaritalStatus_Married",
    "MaritalStatus_Single",
    "AddressGroup_4-6",
    "AddressGroup_7-10",
    "AddressGroup_11+",
    "CouponGroup_1-2",
    "CouponGroup_3-5",
    "CouponGroup_6+",
    "OrderCountGroup_3-5",
    "OrderCountGroup_6-10",
    "OrderCountGroup_11+",
    "OrderAmountHikeGroup_Medium",
    "OrderAmountHikeGroup_High",
    "OrderAmountHikeGroup_Very High",
]


@pytest.fixture
def synthetic_features() -> pd.DataFrame:
    """A small synthetic feature matrix with the same columns as the
    production dataset. Values are random but reproducible.
    """
    rng = np.random.default_rng(42)
    n = 200
    df = pd.DataFrame(
        rng.integers(0, 5, size=(n, len(FEATURE_COLUMNS))),
        columns=FEATURE_COLUMNS,
    )
    return df


@pytest.fixture
def synthetic_labels(synthetic_features: pd.DataFrame) -> pd.Series:
    """Reproducible target that depends on a few real columns so SHAP can find signal."""
    s = (
        (synthetic_features["Tenure"] < 1).astype(int)
        + (synthetic_features["Complain"] > 0).astype(int)
        + (synthetic_features["DaySinceLastOrder"] > 2).astype(int)
    )
    return (s >= 2).astype(int)


@pytest.fixture
def model_bundle(
    tmp_path_factory: pytest.TempPathFactory,
    synthetic_features: pd.DataFrame,
    synthetic_labels: pd.Series,
) -> Path:
    """Train a tiny XGBoost model and write the portable artifact bundle
    (xgboost_model.json + feature_columns.json + metadata.json) to a temp
    directory. Returns the models/ path.
    """
    base = tmp_path_factory.mktemp("model_bundle")
    models = base / "models"
    models.mkdir()

    model = XGBClassifier(
        eval_metric="logloss", random_state=42, n_estimators=20, max_depth=4
    )
    model.fit(synthetic_features, synthetic_labels)
    model.save_model(str(models / "xgboost_model.json"))

    (models / "feature_columns.json").write_text(json.dumps(FEATURE_COLUMNS))
    (models / "metadata.json").write_text(
        json.dumps(
            {
                "model_name": "XGBoost",
                "framework": "xgboost",
                "target_column": "Churn",
                "n_features": len(FEATURE_COLUMNS),
                "n_train_rows": 200,
                "n_test_rows": 0,
                "metrics": {
                    "accuracy": 0.99,
                    "precision": 0.98,
                    "recall": 0.95,
                    "f1": 0.97,
                    "roc_auc": 0.9995,
                },
            }
        )
    )

    # Also write a features.csv in the expected relative location for the
    # explainability tests that read it.
    processed = base / "data" / "processed"
    processed.mkdir(parents=True)
    features_with_target = synthetic_features.copy()
    features_with_target["Churn"] = synthetic_labels.values
    features_with_target.to_csv(processed / "churn_features.csv", index=False)

    return base
