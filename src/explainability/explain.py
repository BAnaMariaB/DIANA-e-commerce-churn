"""
SHAP Explainability Module (Part 3 — Explainable AI)
=====================================================
Loads the trained XGBoost artifact produced by
``src/training/save_artifact.py`` and generates every SHAP visualization the
graded project brief asks for:

- TreeExplainer + Shapley values on the test set.
- Waterfall plot (single point).
- Force plot — single point (saved as HTML, since SHAP's force plots are
  interactive Javascript) AND a static matplotlib version (``matplotlib=True``).
- Force plot — all points at once (HTML).
- Mean |SHAP| bar plot (summary plot with ``plot_type="bar"``).
- Beeswarm plot (full-dataset summary).
- Dependence plots for the top-K most important features.

All outputs are written to ``reports/figures/shap/``.

Run with::

    poetry run python -m src.explainability.explain
    # or for a faster subset on a laptop:
    poetry run python -m src.explainability.explain --sample 200
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless — no display needed (must precede pyplot import)
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import shap  # noqa: E402
import shap.explainers._tree as _shap_tree_mod  # noqa: E402
from xgboost import XGBClassifier  # noqa: E402

# ── XGBoost 3.x ↔ SHAP 0.49 compatibility shim ────────────────────────────────
# XGBoost 3.0+ serializes ``base_score`` as a vector-of-one string (e.g.
# ``"[8.566667E-1]"``) inside the UBJSON model dump. SHAP 0.49.x calls
# ``float()`` on that value directly, which raises ValueError. The fix is
# merged upstream but unreleased. We patch ``float`` only inside SHAP's tree
# loader module so the rest of the code (and our own use of ``float``) is
# untouched. This is a no-op when SHAP is upgraded to a version that
# handles the new format natively.
_BUILTIN_FLOAT = float


def _flex_float(value):  # noqa: D401 — short, single-purpose
    """Drop-in float() that also accepts XGB 3.x ``[scalar]`` string format."""
    if isinstance(value, str) and value.startswith("[") and value.endswith("]"):
        return _BUILTIN_FLOAT(value.strip("[]").split(",")[0])
    return _BUILTIN_FLOAT(value)


_shap_tree_mod.float = _flex_float  # type: ignore[attr-defined]


# ── Paths ─────────────────────────────────────────────────────────────────────

MODELS_DIR = Path("models")
MODEL_FILE = MODELS_DIR / "xgboost_model.json"
COLUMNS_FILE = MODELS_DIR / "feature_columns.json"
METADATA_FILE = MODELS_DIR / "metadata.json"
FEATURES_CSV = Path("data/processed/churn_features.csv")

OUTPUT_DIR = Path("reports/figures/shap")
TARGET_COL = "Churn"
TOP_K_DEPENDENCE = 5  # how many dependence plots to draw


# ── Loaders ───────────────────────────────────────────────────────────────────


def _load_model() -> tuple[XGBClassifier, list[str], dict]:
    """Load the trained XGBoost model + the column list + metadata."""
    if not MODEL_FILE.exists():
        raise FileNotFoundError(
            f"{MODEL_FILE} not found. Run `python -m src.training.save_artifact` first."
        )
    model = XGBClassifier()
    model.load_model(str(MODEL_FILE))
    feature_columns = json.loads(COLUMNS_FILE.read_text())
    metadata = json.loads(METADATA_FILE.read_text())
    return model, feature_columns, metadata


def _load_features(feature_columns: list[str]) -> pd.DataFrame:
    """Load the model-ready feature dataset and drop the target."""
    if not FEATURES_CSV.exists():
        raise FileNotFoundError(
            f"{FEATURES_CSV} not found. Run `python -m src.training.save_artifact` first "
            "(it writes the processed features as a side effect)."
        )
    df = pd.read_csv(FEATURES_CSV)
    if TARGET_COL in df.columns:
        df = df.drop(TARGET_COL, axis=1)
    # Align to the exact training columns + order
    df = df.reindex(columns=feature_columns, fill_value=0)
    return df.astype(int)


# ── Plot helpers ──────────────────────────────────────────────────────────────


def _save_current_fig(path: Path) -> None:
    """Save the active matplotlib figure to ``path`` and close it."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  saved {path}")


def _save_force_plot_html(force_plot, path: Path) -> None:
    """Persist a SHAP interactive force plot as a standalone HTML file."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    shap.save_html(str(path), force_plot)
    print(f"  saved {path}")


# ── Main routine ──────────────────────────────────────────────────────────────


def run_shap_analysis(sample_size: int | None = None) -> None:
    """Compute SHAP values and write every required plot to ``OUTPUT_DIR``.

    Args:
        sample_size: If provided, restricts the analysis to a random sample of
            ``sample_size`` rows. Useful on laptops because the all-points
            force plot becomes very heavy on the full dataset.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Writing outputs to {OUTPUT_DIR.resolve()}\n")

    model, feature_columns, metadata = _load_model()
    print(
        f"Loaded model: {metadata.get('model_name')} ({len(feature_columns)} features)"
    )

    X = _load_features(feature_columns)
    if sample_size is not None and sample_size < len(X):
        X = X.sample(sample_size, random_state=42).reset_index(drop=True)
    print(f"Explaining {len(X)} rows\n")

    # ── 1. TreeExplainer + Shapley values ────────────────────────────────────
    print("[1/8] Building TreeExplainer and computing Shapley values …")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X)  # shap.Explanation object

    # Persist raw values so a downstream notebook can pick them up
    np.save(OUTPUT_DIR / "shap_values.npy", shap_values.values)
    np.save(OUTPUT_DIR / "expected_value.npy", np.array(explainer.expected_value))
    print(f"  saved {OUTPUT_DIR / 'shap_values.npy'}")
    print(f"  saved {OUTPUT_DIR / 'expected_value.npy'}\n")

    # ── 2. Explanation for a SPECIFIC point — waterfall ──────────────────────
    print("[2/8] Waterfall plot for one row …")
    row_idx = 0
    shap.plots.waterfall(shap_values[row_idx], show=False, max_display=15)
    _save_current_fig(OUTPUT_DIR / f"waterfall_row_{row_idx}.png")
    print()

    # ── 3. Force plot — SINGLE point (HTML + static PNG) ─────────────────────
    print("[3/8] Force plot for one row (HTML + static PNG) …")
    # initjs() needs IPython and is only relevant for inline notebook display;
    # save_html() works without it. Try it but tolerate environments without IPython.
    try:
        shap.initjs()
    except Exception:
        pass

    single_force = shap.plots.force(
        explainer.expected_value,
        shap_values.values[row_idx],
        X.iloc[row_idx],
        feature_names=feature_columns,
    )
    _save_force_plot_html(single_force, OUTPUT_DIR / f"force_row_{row_idx}.html")

    # Static (matplotlib) version of the same force plot
    shap.plots.force(
        explainer.expected_value,
        shap_values.values[row_idx],
        X.iloc[row_idx],
        feature_names=feature_columns,
        matplotlib=True,
        show=False,
    )
    _save_current_fig(OUTPUT_DIR / f"force_row_{row_idx}.png")
    print()

    # ── 4. Explanations for ALL points at once — force plot HTML ─────────────
    print("[4/8] Force plot for all points (HTML) …")
    all_force = shap.plots.force(
        explainer.expected_value,
        shap_values.values,
        X,
        feature_names=feature_columns,
    )
    _save_force_plot_html(all_force, OUTPUT_DIR / "force_all_points.html")
    print()

    # ── 5. Summary plot — full dataset, default beeswarm ─────────────────────
    print("[5/8] Beeswarm (summary) plot — full dataset …")
    shap.summary_plot(shap_values.values, X, feature_names=feature_columns, show=False)
    _save_current_fig(OUTPUT_DIR / "beeswarm.png")
    print()

    # ── 6. Mean |SHAP| bar plot ──────────────────────────────────────────────
    print("[6/8] Mean |SHAP| bar plot …")
    shap.summary_plot(
        shap_values.values,
        X,
        feature_names=feature_columns,
        plot_type="bar",
        show=False,
    )
    _save_current_fig(OUTPUT_DIR / "mean_shap_bar.png")
    print()

    # ── 7. Summary plot for each class ───────────────────────────────────────
    # XGBoost binary classifier with shap.TreeExplainer returns a single
    # Explanation tensor whose values represent the log-odds shift toward the
    # positive class. To produce a "per-class" view we plot the same SHAP
    # tensor and its negation so reviewers see contributions for class 0 vs.
    # class 1 side by side (mirrors what shap does internally for multi-class
    # tree models).
    print("[7/8] Summary plot per class …")
    for cls, sign in [(1, +1.0), (0, -1.0)]:
        shap.summary_plot(
            shap_values.values * sign,
            X,
            feature_names=feature_columns,
            show=False,
        )
        plt.suptitle(
            f"Class {cls} ({'will churn' if cls == 1 else 'will stay'})", y=1.02
        )
        _save_current_fig(OUTPUT_DIR / f"summary_class_{cls}.png")
    print()

    # ── 8. Dependence plots for top-K features ───────────────────────────────
    print(f"[8/8] Dependence plots for top {TOP_K_DEPENDENCE} features …")
    mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
    top_indices = np.argsort(mean_abs_shap)[::-1][:TOP_K_DEPENDENCE]
    for rank, idx in enumerate(top_indices, start=1):
        feat = feature_columns[idx]
        # SHAP's dependence_plot picks an interaction feature automatically.
        shap.dependence_plot(
            idx,
            shap_values.values,
            X,
            feature_names=feature_columns,
            show=False,
        )
        # Sanitize the filename — column names like "OrderCountGroup_6-10"
        # contain characters that are safe on POSIX but worth normalizing.
        safe_feat = feat.replace("/", "_").replace(" ", "_")
        _save_current_fig(OUTPUT_DIR / f"dependence_{rank:02d}_{safe_feat}.png")
    print()

    # ── Done ─────────────────────────────────────────────────────────────────
    written = sorted(p.name for p in OUTPUT_DIR.iterdir())
    print(f"✅ SHAP analysis complete. {len(written)} files in {OUTPUT_DIR}/:")
    for name in written:
        print(f"   - {name}")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate SHAP explanations for the churn model."
    )
    p.add_argument(
        "--sample",
        type=int,
        default=None,
        help="Randomly subsample N rows before computing SHAP (default: use all rows).",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_shap_analysis(sample_size=args.sample)
