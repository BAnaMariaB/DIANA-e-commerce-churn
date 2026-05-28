# ─────────────────────────────────────────────────────────────────────────────
# DIANA — Churn Prediction API
# Multi-stage build:
#   1. builder — installs deps, runs the training pipeline, exports the
#                portable XGBoost artifact bundle.
#   2. runtime — slim image that only ships the model + FastAPI server.
# Optimized for Azure Container Apps (scale-to-zero, ~1 GiB RAM budget).
# ─────────────────────────────────────────────────────────────────────────────

# ============================ Stage 1 : builder =============================
FROM python:3.12-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /build

# System deps needed by xgboost wheels + openpyxl
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install training-time Python deps
COPY pyproject.toml ./
RUN pip install --no-cache-dir \
        "pandas>=2.2.0" \
        "numpy>=1.26.0" \
        "scikit-learn>=1.5.0" \
        "xgboost>=3.2.0" \
        "openpyxl>=3.1.5"

# Copy source code and raw data needed for training
COPY src/ ./src/
COPY data/raw/ ./data/raw/

# Train the model and write the portable artifact bundle
ENV PYTHONPATH=/build
RUN python -m src.training.save_artifact


# ============================ Stage 2 : runtime =============================
FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PORT=8000

WORKDIR /app

# libgomp1 is required by xgboost at inference time too
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Runtime deps. scikit-learn is needed because XGBClassifier() inherits
# from sklearn's BaseEstimator and imports it at construction time, even
# when only loading a saved model. openpyxl is still skipped (only used
# during training in the builder stage).
RUN pip install --no-cache-dir \
        "fastapi>=0.115.0" \
        "uvicorn[standard]>=0.32.0" \
        "pandas>=2.2.0" \
        "numpy>=1.26.0" \
        "scikit-learn>=1.5.0" \
        "xgboost>=3.2.0" \
        "pydantic>=2.9.0"

# Copy the inference code and the trained artifact bundle from the builder
COPY src/__init__.py ./src/__init__.py
COPY src/inference/ ./src/inference/
COPY --from=builder /build/models/ ./models/

EXPOSE 8000

# Azure Container Apps injects $PORT; default to 8000 locally.
CMD ["sh", "-c", "uvicorn src.inference.predict:app --host 0.0.0.0 --port ${PORT:-8000}"]
