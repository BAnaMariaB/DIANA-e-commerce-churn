"""Tests for src/inference/predict.py — the FastAPI inference server.

Uses the synthetic model bundle fixture so these run in seconds.
Verifies the happy paths (raw customer + pre-encoded vector), the
error paths (400 on bad payloads), and the API contract (response shape).
"""

from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from tests.conftest import FEATURE_COLUMNS


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch, model_bundle: Path) -> TestClient:
    """Spin up the FastAPI app pointed at the synthetic model bundle."""
    monkeypatch.setenv("MODELS_DIR", str(model_bundle / "models"))
    # Force a fresh import so the module re-reads MODELS_DIR
    for name in list(sys.modules):
        if name.startswith("src.inference"):
            del sys.modules[name]
    from src.inference import predict as predict_mod  # noqa: WPS433

    importlib.reload(predict_mod)
    return TestClient(predict_mod.app)


def test_health(client: TestClient):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_root_returns_links(client: TestClient):
    r = client.get("/")
    assert r.status_code == 200
    body = r.json()
    assert "predict" in body and "health" in body and "info" in body


def test_info_returns_metadata(client: TestClient):
    r = client.get("/info")
    assert r.status_code == 200
    body = r.json()
    assert body["model_name"] == "XGBoost"
    assert body["n_features"] == len(FEATURE_COLUMNS)
    assert "metrics" in body and "roc_auc" in body["metrics"]


def test_predict_raw_customer_payload(client: TestClient):
    payload = {
        "customers": [
            {
                "Tenure": 1,
                "PreferredLoginDevice": "Mobile",
                "CityTier": 3,
                "WarehouseToHome": 12,
                "PreferredPaymentMode": "Debit Card",
                "Gender": "Female",
                "HourSpendOnApp": 4,
                "NumberOfDeviceRegistered": 4,
                "PreferedOrderCat": "Mobile",
                "SatisfactionScore": 1,
                "MaritalStatus": "Single",
                "NumberOfAddress": 9,
                "Complain": 1,
                "OrderAmountHikeFromlastYear": 12,
                "CouponUsed": 0,
                "OrderCount": 1,
                "DaySinceLastOrder": 8,
                "CashbackAmount": 120.0,
                "AddressGroup": "7-10",
                "CouponGroup": "0",
                "OrderCountGroup": "1-2",
                "OrderAmountHikeGroup": "Low",
            }
        ]
    }
    r = client.post("/predict", json=payload)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["model_name"] == "XGBoost"
    assert body["n_features"] == len(FEATURE_COLUMNS)
    assert len(body["predictions"]) == 1
    assert body["predictions"][0] in (0, 1)
    assert 0.0 <= body["probabilities"][0] <= 1.0


def test_predict_encoded_payload(client: TestClient):
    encoded_row = {c: 0 for c in FEATURE_COLUMNS}
    encoded_row["Tenure"] = 1
    encoded_row["Complain"] = 1
    payload = {"features": [encoded_row]}
    r = client.post("/predict", json=payload)
    assert r.status_code == 200, r.text
    body = r.json()
    assert len(body["predictions"]) == 1


def test_predict_rejects_both_fields(client: TestClient):
    r = client.post(
        "/predict",
        json={"customers": [{"Tenure": 1}], "features": [{"Tenure": 1}]},
    )
    assert r.status_code == 400


def test_predict_rejects_empty_body(client: TestClient):
    r = client.post("/predict", json={})
    assert r.status_code == 400


def test_predict_rejects_empty_list(client: TestClient):
    r = client.post("/predict", json={"customers": []})
    assert r.status_code == 400


def test_predict_batch_returns_array(client: TestClient):
    encoded_rows = []
    for tenure in (0, 1, 5, 10):
        row = {c: 0 for c in FEATURE_COLUMNS}
        row["Tenure"] = tenure
        encoded_rows.append(row)
    r = client.post("/predict", json={"features": encoded_rows})
    assert r.status_code == 200
    body = r.json()
    assert len(body["predictions"]) == 4
    assert len(body["probabilities"]) == 4
