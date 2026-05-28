"""
Churn Prediction Inference Server
==================================
FastAPI app that loads the portable XGBoost artifact produced by
``src/training/save_artifact.py`` and exposes:

- ``GET  /health``  — liveness probe for Azure Container Apps.
- ``GET  /info``    — model metadata and training metrics.
- ``GET  /``        — root, returns a short description (avoids 404 for
  the Container App default page health checks).
- ``POST /predict`` — single-customer or batch churn prediction.

The server accepts payloads in two shapes:

1. **Raw customer dict** — the same column names the prep + feature
   engineering pipeline expects (e.g. ``Tenure``, ``PreferredLoginDevice``,
   ``Gender``, …). The server runs the same one-hot encoding the training
   pipeline used so callers don't need to know about the encoded feature
   space.

2. **Pre-encoded vector** — already-encoded columns matching
   ``feature_columns.json``. Used for the regression test suite and by
   internal callers that already have feature vectors on hand.

Run locally::

    poetry run uvicorn src.inference.predict:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from xgboost import XGBClassifier

# ── Paths & config ────────────────────────────────────────────────────────────

MODELS_DIR = Path(os.getenv("MODELS_DIR", "models"))
MODEL_FILE = MODELS_DIR / "xgboost_model.json"
COLUMNS_FILE = MODELS_DIR / "feature_columns.json"
METADATA_FILE = MODELS_DIR / "metadata.json"


# ── Load artifacts on import ──────────────────────────────────────────────────


def _load_artifacts() -> tuple[XGBClassifier, list[str], dict[str, Any]]:
    """Load the trained model, feature columns, and metadata from disk.

    Raises:
        FileNotFoundError: If the artifact bundle is missing. The Docker
            build is responsible for running ``save_artifact.py`` so this
            should only happen during local dev.
    """
    if not MODEL_FILE.exists():
        raise FileNotFoundError(
            f"Model file {MODEL_FILE} not found. "
            "Run `python -m src.training.save_artifact` first."
        )

    model = XGBClassifier()
    model.load_model(str(MODEL_FILE))

    feature_columns = json.loads(COLUMNS_FILE.read_text())
    metadata = json.loads(METADATA_FILE.read_text())

    return model, feature_columns, metadata


MODEL, FEATURE_COLUMNS, METADATA = _load_artifacts()


# ── Schemas ───────────────────────────────────────────────────────────────────


class PredictRequest(BaseModel):
    """Single or batch prediction request.

    Provide *either* ``customers`` (raw, human-readable rows) *or*
    ``features`` (already-encoded vectors).
    """

    customers: list[dict[str, Any]] | None = Field(
        default=None,
        description="List of raw customer rows with the original column names.",
    )
    features: list[dict[str, float | int]] | None = Field(
        default=None,
        description="List of pre-encoded feature vectors matching feature_columns.json.",
    )


class PredictResponse(BaseModel):
    """Prediction response: per-row churn label and probability."""

    predictions: list[int]
    probabilities: list[float]
    model_name: str
    n_features: int


# ── Encoding ──────────────────────────────────────────────────────────────────


def _encode_raw_customers(rows: list[dict[str, Any]]) -> pd.DataFrame:
    """Run the same one-hot encoding as the training pipeline and align
    against the model's expected feature columns.
    """
    df = pd.DataFrame(rows)
    df = pd.get_dummies(df, drop_first=True)
    # Add any missing columns as 0, drop any extras, preserve order.
    df = df.reindex(columns=FEATURE_COLUMNS, fill_value=0)
    return df.astype(int)


def _align_encoded(rows: list[dict[str, float | int]]) -> pd.DataFrame:
    """Align an already-encoded payload to the model's column order.

    Missing columns are filled with 0 (treated as absent one-hot levels).
    """
    df = pd.DataFrame(rows)
    df = df.reindex(columns=FEATURE_COLUMNS, fill_value=0)
    return df.astype(int)


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="DIANA - Churn Prediction API",
    description=(
        "MLflow / XGBoost churn prediction service deployed on Azure "
        "Container Apps. Part of the EPITA AI Project Methodology graded "
        "project (2025-2026)."
    ),
    version=METADATA.get("model_name", "1.0.0"),
)


@app.get("/")
def root() -> dict[str, str]:
    return {
        "service": "DIANA churn prediction",
        "docs": "/docs",
        "health": "/health",
        "info": "/info",
        "predict": "POST /predict",
    }


@app.get("/health")
def health() -> dict[str, str]:
    """Liveness probe — returns 200 OK if the model loaded."""
    return {"status": "ok"}


@app.get("/info")
def info() -> dict[str, Any]:
    """Return model metadata and training metrics."""
    return METADATA


@app.post("/predict", response_model=PredictResponse)
def predict(payload: PredictRequest) -> PredictResponse:
    """Predict churn for one or more customers.

    Returns the binary class (1 = will churn) and the probability.
    """
    if payload.customers and payload.features:
        raise HTTPException(
            status_code=400,
            detail="Provide either 'customers' or 'features', not both.",
        )

    if payload.customers:
        X = _encode_raw_customers(payload.customers)
    elif payload.features:
        X = _align_encoded(payload.features)
    else:
        raise HTTPException(
            status_code=400,
            detail="Request body must contain 'customers' or 'features'.",
        )

    if X.empty:
        raise HTTPException(status_code=400, detail="Empty input.")

    preds = MODEL.predict(X).tolist()
    probas = MODEL.predict_proba(X)[:, 1].tolist()

    return PredictResponse(
        predictions=[int(p) for p in preds],
        probabilities=[float(p) for p in probas],
        model_name=METADATA.get("model_name", "XGBoost"),
        n_features=len(FEATURE_COLUMNS),
    )
