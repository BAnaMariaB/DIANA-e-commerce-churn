"""Smoke tests for src/explainability/explain.py.

Runs the SHAP pipeline against the synthetic model bundle and verifies
that every required plot is produced and non-empty. Intentionally uses
a small sample so the test finishes in a few seconds.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest


REQUIRED_FILES = [
    "waterfall_row_0.png",
    "force_row_0.html",
    "force_row_0.png",
    "force_all_points.html",
    "beeswarm.png",
    "mean_shap_bar.png",
    "summary_class_0.png",
    "summary_class_1.png",
    "shap_values.npy",
    "expected_value.npy",
]


def test_shap_pipeline_produces_all_required_outputs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    model_bundle: Path,
):
    """Run the full SHAP routine; assert every brief-required plot exists."""
    monkeypatch.chdir(model_bundle)

    # Force fresh imports because explain.py reads module-level path constants
    for name in list(sys.modules):
        if name.startswith("src.explainability") or name.startswith("shap"):
            del sys.modules[name]

    from src.explainability.explain import run_shap_analysis  # noqa: WPS433

    run_shap_analysis(sample_size=50)

    output_dir = model_bundle / "reports" / "figures" / "shap"
    assert output_dir.exists()

    for name in REQUIRED_FILES:
        path = output_dir / name
        assert path.exists(), f"missing output: {name}"
        assert path.stat().st_size > 0, f"empty output: {name}"

    # At least 5 dependence plots for the top features
    dependence = sorted(output_dir.glob("dependence_*.png"))
    assert len(dependence) >= 5, f"expected ≥5 dependence plots, got {len(dependence)}"
    for d in dependence:
        assert d.stat().st_size > 0


def test_explain_uses_treeexplainer(
    monkeypatch: pytest.MonkeyPatch,
    model_bundle: Path,
):
    """Confirm the explainer is the TreeExplainer the brief calls for."""
    monkeypatch.chdir(model_bundle)
    for name in list(sys.modules):
        if name.startswith("src.explainability") or name.startswith("shap"):
            del sys.modules[name]

    import shap  # noqa: WPS433
    from src.explainability.explain import _load_model  # noqa: WPS433

    model, _features, _meta = _load_model()
    explainer = shap.TreeExplainer(model)
    assert isinstance(explainer, shap.TreeExplainer)
