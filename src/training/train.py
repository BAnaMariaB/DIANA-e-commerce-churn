"""
Model Training Module
======================
Trains and compares three classification models for churn prediction:
- Logistic Regression (baseline)
- Random Forest
- XGBoost

Uses MLflow to track:
- Parameters
- Metrics (accuracy, precision, recall, f1, roc-auc)
- Models (via MLflow Model Registry)
"""

import pandas as pd
import mlflow
import mlflow.sklearn
import mlflow.xgboost

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)
from xgboost import XGBClassifier


# ── Config ────────────────────────────────────────────────────────────────────

DATA_PATH = "data/processed/churn_features.csv"
TARGET_COL = "Churn"
TEST_SIZE = 0.2
RANDOM_STATE = 42
MLFLOW_EXPERIMENT = "churn-prediction"


# ── Data loading ──────────────────────────────────────────────────────────────


def load_features(filepath: str):
    """Load model-ready feature dataset.

    Args:
        filepath: Path to the encoded CSV.

    Returns:
        Tuple (X, y) of features and target.
    """
    df = pd.read_csv(filepath)
    X = df.drop(TARGET_COL, axis=1)
    y = df[TARGET_COL]
    print(f"Features: {X.shape}, Target: {y.shape}")
    return X, y


# ── Evaluation ────────────────────────────────────────────────────────────────


def evaluate_model(model_name: str, y_test, y_pred, y_proba) -> dict:
    """Compute and print evaluation metrics.

    Args:
        model_name: Name of the model for display.
        y_test: True labels.
        y_pred: Predicted labels.
        y_proba: Predicted probabilities for positive class.

    Returns:
        Dictionary of metric name → value.
    """
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba),
    }

    print(f"\n{'='*60}")
    print(f"  {model_name}")
    print(f"{'='*60}")
    for k, v in metrics.items():
        print(f"  {k:<12}: {v:.4f}")
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    return metrics


# ── Training ──────────────────────────────────────────────────────────────────


def train_all_models(X, y) -> pd.DataFrame:
    """Train Logistic Regression, Random Forest, and XGBoost with MLflow tracking.

    Args:
        X: Feature matrix.
        y: Target vector.

    Returns:
        DataFrame summarizing all model results.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    # Scale only for Logistic Regression
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    models = {
        "Logistic Regression": (
            LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
            True,  # needs scaling
        ),
        "Random Forest": (
            RandomForestClassifier(
                n_estimators=200, random_state=RANDOM_STATE, class_weight="balanced"
            ),
            False,
        ),
        "XGBoost": (
            XGBClassifier(eval_metric="logloss", random_state=RANDOM_STATE),
            False,
        ),
    }

    results = []

    for name, (model, scaled) in models.items():
        with mlflow.start_run(run_name=name):

            # ── Train ──
            if scaled:
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
                y_proba = model.predict_proba(X_test_scaled)[:, 1]
            else:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                y_proba = model.predict_proba(X_test)[:, 1]

            # ── Evaluate ──
            metrics = evaluate_model(name, y_test, y_pred, y_proba)

            # ── Log to MLflow ──
            mlflow.log_param("model_name", name)
            mlflow.log_param("test_size", TEST_SIZE)
            mlflow.log_param("random_state", RANDOM_STATE)
            mlflow.log_metrics(metrics)

            # Log model
            if name == "XGBoost":
                mlflow.xgboost.log_model(
                    model, artifact_path="model", registered_model_name=name
                )
            else:
                mlflow.sklearn.log_model(
                    model, artifact_path="model", registered_model_name=name
                )

            results.append({"Model": name, **metrics})

    results_df = pd.DataFrame(results).sort_values("roc_auc", ascending=False)

    print("\n" + "=" * 60)
    print("FINAL MODEL COMPARISON")
    print("=" * 60)
    print(results_df.to_string(index=False))

    return results_df


if __name__ == "__main__":
    X, y = load_features(DATA_PATH)
    results = train_all_models(X, y)
