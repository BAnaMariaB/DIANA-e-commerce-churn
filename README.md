# DIANA - E-Commerce Customer Churn Prediction

**EPITA - AI Project Methodology 2025-2026**

## Project Overview

This project builds a production-ready ML pipeline to predict customer churn for an e-commerce company, following AI project best practices. It uses behavioral and transactional data to identify customers at risk of leaving the platform.

**Best model: XGBoost — Accuracy: 99%, ROC-AUC: 0.9995**

## Project Structure

```
DIANA-e-commerce-churn/
├── data/
│   ├── raw/                        # Original dataset (not versioned)
│   ├── processed/                  # Cleaned and encoded data (not versioned)
│   └── external/                   # External data sources
├── notebooks/                      # EDA and exploration
├── src/
│   ├── data_preparation/
│   │   ├── prepare.py              # Data loading, cleaning, imputation
│   │   └── statistical_analysis.py # T-test, chi-square, Mann-Whitney
│   ├── feature_engineering/
│   │   └── features.py             # Feature transformation and encoding
│   ├── training/
│   │   └── train.py                # Model training + MLflow tracking
│   └── inference/
│       └── predict.py              # Model serving and predictions
├── models/                         # Saved model artifacts
├── docs/                           # Sphinx documentation
├── mlruns/                         # MLflow experiment outputs
├── tests/                          # Unit tests
├── MLproject                       # MLflow project configuration
├── python_env.yaml                 # MLflow environment specification
├── pyproject.toml                  # Poetry dependency management
└── .flake8                         # Flake8 configuration
```

## Models Compared

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| XGBoost | 99.0% | 98.4% | 95.8% | 97.1% | 0.9995 |
| Random Forest | 97.8% | 99.4% | 87.4% | 93.0% | 0.9991 |
| Logistic Regression | 88.7% | 72.3% | 53.7% | 61.6% | 0.8850 |

## Setup

### Requirements
- Python 3.12
- Poetry 2.4+
- libomp (Mac only, for XGBoost): `brew install libomp`

### Installation

```bash
# Clone the repository
git clone <repo-url>
cd DIANA-e-commerce-churn

# Install dependencies
poetry install

# Activate the environment
poetry shell
```

## Usage

### Run the full pipeline

```bash
# Step 1 - Data preparation
poetry run python src/data_preparation/prepare.py

# Step 2 - Statistical analysis
poetry run python src/data_preparation/statistical_analysis.py

# Step 3 - Feature engineering
poetry run python src/feature_engineering/features.py

# Step 4 - Model training
poetry run python src/training/train.py

# Step 5 - Launch MLflow UI
poetry run mlflow ui
# Open http://127.0.0.1:5000
```

### Run with MLflow Projects

```bash
poetry run mlflow run . -e train --env-manager=local --experiment-name=churn-prediction
```

### Serve the model locally

```bash
poetry run mlflow models serve -m "models:/XGBoost/1" --port 5001 --no-conda
```

### View documentation

```bash
poetry run sphinx-build docs/source docs/build
open docs/build/index.html
```

### Code quality

```bash
# Auto-format
poetry run black src/

# Check PEP8
poetry run flake8 src/
```

## Tools & Best Practices

| Requirement | Tool |
|---|---|
| Dependency management | Poetry |
| Code style (PEP8) | Black, Flake8, Pylint |
| Experiment tracking | MLflow |
| Model registry | MLflow Model Registry |
| Model serving | MLflow serving |
| Reproducibility | MLflow Projects |
| Documentation | Sphinx |
| Testing | Pytest |
| Versioning | Git + GitHub |

## Dataset

[E-Commerce Churn Dataset](https://www.kaggle.com/datasets/ankitverma2010/ecommerce-customer-churn-analysis-and-prediction)

## Authors

- Ana-Maria Borduselu
- Kruthi Shandilya
- Ran Navlani

EPITA - AI Project Methodology 2025-2026