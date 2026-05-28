"""Unit tests for src/feature_engineering/features.py.

Covers label normalization, grouped feature creation, ID drop, and the
shape of the final one-hot encoded matrix.
"""

from __future__ import annotations

import pandas as pd

from src.feature_engineering.features import (
    clean_categorical_labels,
    create_grouped_features,
    drop_id_column,
    encode_features,
)


def _sample_df() -> pd.DataFrame:
    """A small mock with the inconsistencies the real dataset has."""
    return pd.DataFrame(
        {
            "CustomerID": [1, 2, 3, 4],
            "PreferredLoginDevice": ["Mobile Phone", "Phone", "Computer", "Mobile"],
            "PreferredPaymentMode": ["CC", "Credit Card", "COD", "UPI"],
            "PreferedOrderCat": ["Mobile Phone", "Grocery", "Laptop & Accessory", "Others"],
            "NumberOfAddress": [2, 5, 8, 12],
            "CouponUsed": [0, 1, 3, 7],
            "OrderCount": [1, 3, 7, 15],
            "OrderAmountHikeFromlastYear": [10, 14, 18, 25],
        }
    )


def test_clean_categorical_labels_normalizes_known_typos():
    df = clean_categorical_labels(_sample_df())
    assert set(df["PreferredLoginDevice"].unique()) <= {"Mobile", "Computer"}
    assert "CC" not in df["PreferredPaymentMode"].unique()
    assert "Credit Card" in df["PreferredPaymentMode"].unique()
    assert "Cash on Delivery" in df["PreferredPaymentMode"].unique()
    assert "Mobile Phone" not in df["PreferedOrderCat"].unique()


def test_create_grouped_features_adds_expected_bins():
    df = create_grouped_features(_sample_df())
    for col in ("AddressGroup", "CouponGroup", "OrderCountGroup", "OrderAmountHikeGroup"):
        assert col in df.columns, f"{col} missing"
    # Spot-check a couple of bins
    assert str(df["AddressGroup"].iloc[0]) == "1-3"
    assert str(df["AddressGroup"].iloc[3]) == "11+"
    assert str(df["OrderAmountHikeGroup"].iloc[0]) == "Low"
    assert str(df["OrderAmountHikeGroup"].iloc[3]) == "Very High"


def test_drop_id_column_removes_customer_id():
    df = drop_id_column(_sample_df())
    assert "CustomerID" not in df.columns


def test_drop_id_column_is_noop_when_absent():
    df = _sample_df().drop("CustomerID", axis=1)
    out = drop_id_column(df)
    pd.testing.assert_frame_equal(out, df)


def test_encode_features_produces_integer_matrix():
    df = pd.DataFrame(
        {
            "Tenure": [1, 2, 3],
            "Gender": ["Male", "Female", "Female"],
            "Churn": [0, 1, 0],
        }
    )
    out = encode_features(df)
    # All columns numeric (int)
    assert all(pd.api.types.is_integer_dtype(out[c]) for c in out.columns)
    # drop_first=True means Gender_Male should remain but Gender_Female should not
    assert "Gender_Male" in out.columns
    assert "Gender_Female" not in out.columns
