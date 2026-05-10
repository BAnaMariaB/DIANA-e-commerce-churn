# DIANA E-Commerce Customer Churn Prediction

EPITA - AI Project Methodology 2025-2026

## Project Overview
This project builds a production-ready ML pipeline to predict customer churn 
for an e-commerce company, following AI project best practices.

## Project Structure 
churn-prediction/
├── data/
│   ├── raw/                # Original dataset (not versioned)
│   ├── processed/          # Cleaned data (not versioned)
│   └── external/           # External data sources
├── notebooks/              # EDA and exploration
├── src/
│   ├── data_preparation/   # Data loading and cleaning
│   ├── feature_engineering/ # Feature transformations
│   ├── training/           # Model training + MLflow tracking
│   └── inference/          # Model serving
├── models/                 # Saved model artifacts
├── docs/                   # Sphinx documentation
├── tests/                  # Unit tests
└── mlruns/                 # MLflow experiment outputs## Setup

### Requirements
- Python 3.12
- Poetry 2.4+

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

```bash
# Data preparation
poetry run python src/data_preparation/prepare.py

# Feature engineering
poetry run python src/feature_engineering/features.py

# Model training
poetry run python src/training/train.py

# Launch MLflow UI
poetry run mlflow ui
```

## Tools & Best Practices
- **Dependency management**: Poetry
- **Code style**: Black, Flake8, Pylint (PEP8)
- **Experiment tracking**: MLflow
- **Documentation**: Sphinx
- **Testing**: Pytest
- **Versioning**: Git + GitHub

## Dataset
[E-Commerce Churn Dataset](https://www.kaggle.com/datasets/ankitverma2010/ecommerce-customer-churn-analysis-and-prediction)

## Authors
- Group members: ...
