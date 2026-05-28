"""Tests that the portable artifact bundle is loadable and self-consistent.

The Docker image trains the model during the builder stage and then ships
the resulting bundle to the runtime image. If the bundle layout ever
drifts, the deployed container fails to start — this test catches that.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from tests.conftest import FEATURE_COLUMNS


def test_bundle_files_exist(model_bundle: Path):
    """All three files in the bundle must be present and non-empty."""
    for name in ("xgboost_model.json", "feature_columns.json", "metadata.json"):
        path = model_bundle / "models" / name
        assert path.exists(), f"missing {name}"
        assert path.stat().st_size > 0, f"empty {name}"


def test_bundle_columns_match_feature_columns(model_bundle: Path):
    """The columns the API will validate against must equal the project's canonical list."""
    cols = json.loads((model_bundle / "models" / "feature_columns.json").read_text())
    assert cols == FEATURE_COLUMNS


def test_bundle_metadata_has_required_fields(model_bundle: Path):
    meta = json.loads((model_bundle / "models" / "metadata.json").read_text())
    for key in ("model_name", "framework", "n_features", "metrics"):
        assert key in meta, f"missing metadata key: {key}"
    for metric in ("accuracy", "precision", "recall", "f1", "roc_auc"):
        assert metric in meta["metrics"], f"missing metric: {metric}"


def test_bundle_model_loads_and_predicts(model_bundle: Path):
    """Loading the saved JSON model and running predict_proba must succeed."""
    model = XGBClassifier()
    model.load_model(str(model_bundle / "models" / "xgboost_model.json"))

    X = pd.DataFrame(
        np.zeros((1, len(FEATURE_COLUMNS)), dtype=int), columns=FEATURE_COLUMNS
    )
    proba = model.predict_proba(X)[:, 1]
    assert proba.shape == (1,)
    assert 0.0 <= float(proba[0]) <= 1.0
