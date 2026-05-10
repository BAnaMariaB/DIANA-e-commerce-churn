"""
Statistical Analysis Module
=============================
Runs statistical tests to identify features significantly
associated with customer churn:
- T-Test (numerical features)
- Chi-Square Test (categorical features)
- Mann-Whitney U Test (non-parametric numerical)
- Correlation analysis
"""

import pandas as pd
from scipy.stats import ttest_ind, chi2_contingency, mannwhitneyu


# ── Column definitions ────────────────────────────────────────────────────────

NUMERICAL_TEST_COLS = ["Tenure", "WarehouseToHome", "DaySinceLastOrder", "CashbackAmount"]

CATEGORICAL_TEST_COLS = [
    "PreferredPaymentMode",
    "PreferedOrderCat",
    "MaritalStatus",
    "Complain",
]

MANN_WHITNEY_COLS = ["CashbackAmount", "DaySinceLastOrder", "WarehouseToHome", "OrderCount"]


# ── Statistical tests ─────────────────────────────────────────────────────────

def run_ttest(df: pd.DataFrame) -> None:
    """Run independent t-tests for numerical features vs Churn.

    Args:
        df: DataFrame with numerical features and Churn column.
    """
    print("\n" + "=" * 60)
    print("T-TEST RESULTS (Numerical Features)")
    print("=" * 60)

    for col in NUMERICAL_TEST_COLS:
        churn_0 = df[df["Churn"] == 0][col].dropna()
        churn_1 = df[df["Churn"] == 1][col].dropna()
        stat, p = ttest_ind(churn_0, churn_1)

        print(f"\n  {col}")
        print(f"  T-statistic : {stat:.4f}")
        print(f"  P-value     : {p:.6f}")
        print(f"  → {'Significant difference' if p < 0.05 else 'No significant difference'}")


def run_chi2(df: pd.DataFrame) -> None:
    """Run chi-square tests for categorical features vs Churn.

    Args:
        df: DataFrame with categorical features and Churn column.
    """
    print("\n" + "=" * 60)
    print("CHI-SQUARE TEST RESULTS (Categorical Features)")
    print("=" * 60)

    for col in CATEGORICAL_TEST_COLS:
        contingency_table = pd.crosstab(df[col], df["Churn"])
        chi2, p, dof, _ = chi2_contingency(contingency_table)

        print(f"\n  {col}")
        print(f"  Chi-square : {chi2:.4f}")
        print(f"  P-value    : {p:.6f}")
        print(f"  → {'Significant association with churn' if p < 0.05 else 'No significant association'}")


def run_mannwhitney(df: pd.DataFrame) -> None:
    """Run Mann-Whitney U tests for non-parametric comparison.

    Args:
        df: DataFrame with numerical features and Churn column.
    """
    print("\n" + "=" * 60)
    print("MANN-WHITNEY U TEST RESULTS")
    print("=" * 60)

    for col in MANN_WHITNEY_COLS:
        churn_0 = df[df["Churn"] == 0][col].dropna()
        churn_1 = df[df["Churn"] == 1][col].dropna()
        stat, p = mannwhitneyu(churn_0, churn_1)

        print(f"\n  {col}")
        print(f"  U-statistic : {stat:.4f}")
        print(f"  P-value     : {p:.6f}")
        print(f"  → {'Significant distribution difference' if p < 0.05 else 'No significant difference'}")


def run_correlation(df: pd.DataFrame) -> None:
    """Compute and display correlation of all features with Churn.

    Args:
        df: DataFrame with numerical features and Churn column.
    """
    print("\n" + "=" * 60)
    print("CORRELATION WITH CHURN")
    print("=" * 60)

    corr = df.corr(numeric_only=True)["Churn"].sort_values(ascending=False)
    print(corr.to_string())


def run_all_tests(filepath: str) -> None:
    """Run all statistical tests on the cleaned dataset.

    Args:
        filepath: Path to cleaned CSV (before encoding).
    """
    df = pd.read_csv(filepath)

    run_ttest(df)
    run_chi2(df)
    run_mannwhitney(df)
    run_correlation(df)


if __name__ == "__main__":
    CLEANED_PATH = "data/processed/churn_cleaned.csv"
    run_all_tests(CLEANED_PATH)