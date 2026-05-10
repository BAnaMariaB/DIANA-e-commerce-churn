"""
Feature Engineering Module
===========================
Transforms the cleaned dataset:
- Standardizes categorical labels (duplicates / typos)
- Creates grouped/binned features
- Imputes remaining missing values
- Drops non-predictive columns (CustomerID)
- One-hot encodes categorical features
- Saves model-ready dataset to data/processed/
"""

import pandas as pd


# ── Column definitions ────────────────────────────────────────────────────────

NUMERICAL_COLS = ["Tenure", "WarehouseToHome", "DaySinceLastOrder", "CashbackAmount"]

CATEGORICAL_COLS = [
    "PreferredLoginDevice",
    "CityTier",
    "PreferredPaymentMode",
    "Gender",
    "PreferedOrderCat",
    "MaritalStatus",
    "NumberOfDeviceRegistered",
    "SatisfactionScore",
    "NumberOfAddress",
    "OrderAmountHikeFromlastYear",
    "CouponUsed",
    "OrderCount",
    "Complain",
]


# ── Cleaning helpers ──────────────────────────────────────────────────────────


def clean_categorical_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize inconsistent category labels.

    Args:
        df: Cleaned DataFrame from data preparation.

    Returns:
        DataFrame with standardized labels.
    """
    df["PreferredLoginDevice"] = df["PreferredLoginDevice"].replace(
        {"Mobile Phone": "Mobile", "Phone": "Mobile"}
    )
    df["PreferredPaymentMode"] = df["PreferredPaymentMode"].replace(
        {"CC": "Credit Card", "COD": "Cash on Delivery"}
    )
    df["PreferedOrderCat"] = df["PreferedOrderCat"].replace({"Mobile Phone": "Mobile"})
    return df


def create_grouped_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create binned/grouped versions of discrete columns.

    Args:
        df: DataFrame with standardized labels.

    Returns:
        DataFrame with added group columns.
    """
    df["AddressGroup"] = pd.cut(
        df["NumberOfAddress"],
        bins=[0, 3, 6, 10, 100],
        labels=["1-3", "4-6", "7-10", "11+"],
    )
    df["CouponGroup"] = pd.cut(
        df["CouponUsed"],
        bins=[-1, 0, 2, 5, 100],
        labels=["0", "1-2", "3-5", "6+"],
    )
    df["OrderCountGroup"] = pd.cut(
        df["OrderCount"],
        bins=[0, 2, 5, 10, 100],
        labels=["1-2", "3-5", "6-10", "11+"],
    )
    df["OrderAmountHikeGroup"] = pd.cut(
        df["OrderAmountHikeFromlastYear"],
        bins=[0, 12, 16, 20, 100],
        labels=["Low", "Medium", "High", "Very High"],
    )
    return df


def impute_remaining_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Impute any remaining missing values before encoding.

    Numerical columns → median. Categorical columns → mode.

    Args:
        df: DataFrame with possible remaining NaNs.

    Returns:
        Fully imputed DataFrame.
    """
    print("Missing values before imputation:")
    print(df.isnull().sum()[df.isnull().sum() > 0])

    # Discrete behavioral columns
    discrete_cols = ["OrderAmountHikeFromlastYear", "CouponUsed", "OrderCount"]
    for col in discrete_cols:
        df[col] = df[col].fillna(df[col].median())

    # All remaining numeric columns
    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns
    for col in numeric_cols:
        df[col] = df[col].fillna(df[col].median())

    # All remaining categorical columns
    cat_cols = df.select_dtypes(include=["object", "category"]).columns
    for col in cat_cols:
        df[col] = df[col].fillna(df[col].mode()[0])

    print("\nMissing values after imputation:")
    print(df.isnull().sum()[df.isnull().sum() > 0])
    return df


def drop_id_column(df: pd.DataFrame) -> pd.DataFrame:
    """Drop the CustomerID column (non-predictive identifier).

    Args:
        df: DataFrame with CustomerID column.

    Returns:
        DataFrame without CustomerID.
    """
    if "CustomerID" in df.columns:
        df = df.drop("CustomerID", axis=1)
        print("Dropped 'CustomerID' column.")
    return df


def encode_features(df: pd.DataFrame) -> pd.DataFrame:
    """One-hot encode all categorical features.

    Args:
        df: DataFrame with categorical columns.

    Returns:
        Fully encoded integer DataFrame ready for modelling.
    """
    df = pd.get_dummies(df, drop_first=True)
    df = df.astype(int)
    print(f"Encoded dataset shape: {df.shape}")
    return df


def run_feature_engineering(input_path: str, output_path: str) -> pd.DataFrame:
    """Full feature engineering pipeline.

    Args:
        input_path: Path to cleaned CSV from data preparation.
        output_path: Path to save model-ready CSV.

    Returns:
        Model-ready DataFrame.
    """
    df = pd.read_csv(input_path)
    df = clean_categorical_labels(df)
    df = create_grouped_features(df)
    df = impute_remaining_missing(df)
    df = drop_id_column(df)
    df = encode_features(df)
    df.to_csv(output_path, index=False)
    print(f"Model-ready data saved to: {output_path}")
    return df


if __name__ == "__main__":
    INPUT_PATH = "data/processed/churn_cleaned.csv"
    OUTPUT_PATH = "data/processed/churn_features.csv"

    run_feature_engineering(INPUT_PATH, OUTPUT_PATH)
