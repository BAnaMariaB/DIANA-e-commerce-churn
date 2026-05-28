"""Unit tests for src/data_preparation/prepare.py.

Covers the cleaning + imputation helpers without touching the real
Kaggle file — we feed crafted DataFrames in.
"""

from __future__ import annotations

import pandas as pd

from src.data_preparation.prepare import impute_missing_numerical


def test_impute_missing_numerical_fills_with_median():
    df = pd.DataFrame(
        {
            "Tenure": [1.0, 2.0, None, 4.0, 5.0],
            "WarehouseToHome": [10.0, None, 30.0, 40.0, 50.0],
            "DaySinceLastOrder": [None, 1.0, 2.0, 3.0, 4.0],
            "Untouched": [100, 200, 300, 400, 500],
        }
    )
    out = impute_missing_numerical(df)
    # No more NaNs in the targeted columns
    assert out["Tenure"].isnull().sum() == 0
    assert out["WarehouseToHome"].isnull().sum() == 0
    assert out["DaySinceLastOrder"].isnull().sum() == 0
    # And the filled value equals the median of the non-null values
    assert out["Tenure"].iloc[2] == 3.0
    assert out["WarehouseToHome"].iloc[1] == 35.0
    assert out["DaySinceLastOrder"].iloc[0] == 2.5
    # Other columns are untouched
    assert list(out["Untouched"]) == [100, 200, 300, 400, 500]


def test_impute_missing_numerical_noop_when_no_missing():
    df = pd.DataFrame(
        {
            "Tenure": [1.0, 2.0, 3.0],
            "WarehouseToHome": [10.0, 20.0, 30.0],
            "DaySinceLastOrder": [1.0, 2.0, 3.0],
        }
    )
    out = impute_missing_numerical(df.copy())
    pd.testing.assert_frame_equal(out, df)
