"""
Customer Churn Prediction System - Phase 1: Data Preparation
Script: explore_data.py

Description:
Performs Exploratory Data Analysis (EDA) and automated data validation
on data/customer_churn.csv.
"""

import os
import sys
import pandas as pd


def load_dataset(file_path: str = "data/customer_churn.csv") -> pd.DataFrame:
    """Load the CSV dataset and handle missing file errors gracefully."""
    if not os.path.exists(file_path):
        print(f"[ERROR] Data file not found at: {file_path}")
        print("Please run `python generate_data.py` first to create the dataset.")
        sys.exit(1)

    df = pd.read_csv(file_path)
    return df


def display_overview(df: pd.DataFrame) -> None:
    """Display basic structure, columns, types, and sample records."""
    print("\n" + "=" * 65)
    print("1. DATASET OVERVIEW")
    print("=" * 65)

    # Dataset shape
    rows, cols = df.shape
    print(f"Dataset Shape: {rows} rows x {cols} columns\n")

    # Column names and data types
    print("Column Information & Data Types:")
    dtype_df = pd.DataFrame({
        "Column Name": df.columns,
        "Data Type": df.dtypes.astype(str).values,
        "Non-Null Count": df.notnull().sum().values
    })
    print(dtype_df.to_string(index=False))

    # Identify Features vs Target
    target_col = "churn"
    feature_cols = [col for col in df.columns if col != target_col]
    print(f"\nTarget Column (Supervised ML Label):")
    print(f"  -> '{target_col}' (0 = customer stays, 1 = customer churns)")
    print(f"Feature Columns (Input Predictors):")
    for feat in feature_cols:
        print(f"  -> '{feat}'")

    # First 10 rows
    print("\nFirst 10 Rows:")
    print(df.head(10).to_string())


def check_data_quality(df: pd.DataFrame) -> None:
    """Check for missing values and duplicate records."""
    print("\n" + "=" * 65)
    print("2. DATA INTEGRITY & QUALITY CHECKS")
    print("=" * 65)

    # Missing values
    missing = df.isnull().sum()
    print("Missing Values Per Column:")
    print(missing.to_string())
    total_missing = missing.sum()
    print(f"Total Missing Values across dataset: {total_missing}")

    # Duplicate rows
    duplicates = df.duplicated().sum()
    print(f"\nDuplicate Rows: {duplicates}")


def display_statistics(df: pd.DataFrame) -> None:
    """Display descriptive statistics, class distribution, and feature correlations."""
    print("\n" + "=" * 65)
    print("3. DESCRIPTIVE STATISTICS & DISTRIBUTIONS")
    print("=" * 65)

    # Summary statistics for numeric features
    print("Summary Statistics (Numeric Features):")
    print(df.describe().T[["count", "mean", "std", "min", "25%", "50%", "75%", "max"]].to_string())

    # Target class distribution
    print("\nTarget Variable ('churn') Class Distribution:")
    counts = df["churn"].value_counts().sort_index()
    percentages = (df["churn"].value_counts(normalize=True).sort_index() * 100).round(2)

    dist_df = pd.DataFrame({
        "Churn Label": ["0 (Stayed)", "1 (Churned)"],
        "Count": counts.values,
        "Percentage (%)": percentages.values
    })
    print(dist_df.to_string(index=False))

    # Correlations between features and target
    print("\nCorrelation with Target ('churn'):")
    corr = df.corr()["churn"].drop("churn").sort_values(ascending=False)
    corr_df = pd.DataFrame({
        "Feature": corr.index,
        "Pearson Correlation": corr.values.round(4)
    })
    print(corr_df.to_string(index=False))


def validate_dataset(df: pd.DataFrame) -> bool:
    """
    Run automated validation checks on dataset constraints and return status.
    """
    print("\n" + "=" * 65)
    print("4. DATA VALIDATION CHECKS")
    print("=" * 65)

    checks_passed = 0
    total_checks = 8
    all_passed = True

    def log_check(check_id: int, description: str, condition: bool, details: str = ""):
        nonlocal checks_passed, all_passed
        if condition:
            checks_passed += 1
            status = "[PASS]"
        else:
            all_passed = False
            status = "[FAIL]"
        print(f"  {status} Check {check_id}: {description} {details}")

    # 1. Missing values check
    missing_count = df.isnull().sum().sum()
    log_check(1, "No missing values in dataset", missing_count == 0, f"(found {missing_count})")

    # 2. Target column validity (churn must be only 0 or 1)
    unique_churn = set(df["churn"].unique())
    valid_churn = unique_churn.issubset({0, 1})
    log_check(2, "Target 'churn' contains only 0 and 1", valid_churn, f"(unique values: {[int(x) for x in sorted(list(unique_churn))]})")

    # 3. tenure_months range [1, 72]
    tenure_valid = (df["tenure_months"] >= 1).all() and (df["tenure_months"] <= 72).all()
    log_check(3, "Feature 'tenure_months' in range [1, 72]", tenure_valid,
              f"(min: {int(df['tenure_months'].min())}, max: {int(df['tenure_months'].max())})")

    # 4. monthly_charge range [20.0, 150.0]
    charge_valid = (df["monthly_charge"] >= 20.0).all() and (df["monthly_charge"] <= 150.0).all()
    log_check(4, "Feature 'monthly_charge' in range [$20.00, $150.00]", charge_valid,
              f"(min: ${df['monthly_charge'].min():.2f}, max: ${df['monthly_charge'].max():.2f})")

    # 5. support_tickets non-negative integer
    tickets_valid = (df["support_tickets"] >= 0).all() and pd.api.types.is_integer_dtype(df["support_tickets"])
    log_check(5, "Feature 'support_tickets' >= 0 integer count", tickets_valid,
              f"(min: {int(df['support_tickets'].min())}, max: {int(df['support_tickets'].max())})")

    # 6. contract_months in {1, 12, 24}
    valid_contracts = set(df["contract_months"].unique()).issubset({1, 12, 24})
    log_check(6, "Feature 'contract_months' in {1, 12, 24}", valid_contracts,
              f"(unique values: {[int(x) for x in sorted(list(df['contract_months'].unique()))]})")

    # 7. autopay in {0, 1}
    valid_autopay = set(df["autopay"].unique()).issubset({0, 1})
    log_check(7, "Feature 'autopay' is binary (0 or 1)", valid_autopay,
              f"(unique values: {[int(x) for x in sorted(list(df['autopay'].unique()))]})")

    # 8. usage_hours range [5.0, 100.0]
    usage_valid = (df["usage_hours"] >= 5.0).all() and (df["usage_hours"] <= 100.0).all()
    log_check(8, "Feature 'usage_hours' in range [5.0, 100.0]", usage_valid,
              f"(min: {df['usage_hours'].min():.1f}, max: {df['usage_hours'].max():.1f})")

    print("-" * 65)
    print(f"Validation Summary: {checks_passed}/{total_checks} checks passed.")
    if all_passed:
        print("OVERALL RESULT: [PASSED] - Dataset is valid and ready for subsequent ML phases.")
    else:
        print("OVERALL RESULT: [FAILED] - Please review validation issues before continuing.")
    print("=" * 65)

    return all_passed


def main():
    csv_path = "data/customer_churn.csv"
    print("=" * 65)
    print("CUSTOMER CHURN PREDICTION SYSTEM - DATA EXPLORATION & VALIDATION")
    print("=" * 65)

    df = load_dataset(csv_path)
    display_overview(df)
    check_data_quality(df)
    display_statistics(df)
    validate_dataset(df)


if __name__ == "__main__":
    main()
