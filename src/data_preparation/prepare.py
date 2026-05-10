"""
Data Preparation Module
=======================
Loads the raw e-commerce dataset and performs initial cleaning:
- Loads the Excel file
- Inspects shape, dtypes, missing values
- Imputes missing numerical values with median
- Saves cleaned data to data/processed/
"""

import pandas as pd


def load_data(filepath: str) -> pd.DataFrame:
    """Load raw dataset from Excel file.

    Args:
        filepath: Path to the Excel file.

    Returns:
        Raw DataFrame.
    """
    df = pd.read_excel(filepath)
    print(f"Dataset loaded: {df.shape[0]} rows x {df.shape[1]} columns")
    return df


def inspect_data(df: pd.DataFrame) -> None:
    """Print basic dataset information.

    Args:
        df: Raw DataFrame.
    """
    print("=" * 67)
    print("FIRST 5 ROWS")
    print("=" * 67)
    print(df.head())

    print("\n" + "=" * 67)
    print("STATISTICAL METRICS")
    print("=" * 67)
    print(df.describe())

    print("\n" + "=" * 67)
    print("SHAPE")
    print("=" * 67)
    print(df.shape)

    print("\n" + "=" * 67)
    print("COLUMN INFO")
    print("=" * 67)
    print(df.info())

    print("\n" + "=" * 67)
    print("MISSING VALUES")
    print("=" * 67)
    print(df.isnull().sum())


def impute_missing_numerical(df: pd.DataFrame) -> pd.DataFrame:
    """Impute missing values in numerical columns using median.

    Args:
        df: DataFrame with possible missing values.

    Returns:
        DataFrame with imputed numerical columns.
    """
    numerical_cols = ["Tenure", "WarehouseToHome", "DaySinceLastOrder"]

    for col in numerical_cols:
        missing = df[col].isnull().sum()
        if missing > 0:
            df[col] = df[col].fillna(df[col].median())
            print(f"Imputed {missing} missing values in '{col}' with median.")

    return df


def save_processed(df: pd.DataFrame, output_path: str) -> None:
    """Save the cleaned DataFrame to CSV.

    Args:
        df: Cleaned DataFrame.
        output_path: Path to save the processed CSV.
    """
    df.to_csv(output_path, index=False)
    print(f"Processed data saved to: {output_path}")


if __name__ == "__main__":
    RAW_PATH = "data/raw/ECommerceDataset2.xlsx"
    PROCESSED_PATH = "data/processed/churn_cleaned.csv"

    df = load_data(RAW_PATH)
    inspect_data(df)
    df = impute_missing_numerical(df)
    save_processed(df, PROCESSED_PATH)
