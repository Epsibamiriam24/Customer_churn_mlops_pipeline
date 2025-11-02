# src/eda_clean.py
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

def eda_and_clean(train_path, test_path=None):
    print("Loading datasets...")
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path) if test_path else None

    print("\n--- Basic Info ---")
    print(train.info())
    print("\n--- First 5 Rows ---")
    print(train.head())

    print("\n--- Missing Values (%) ---")
    missing = (train.isna().sum() / len(train)) * 100
    print(missing[missing > 0])

    # Drop completely empty rows
    train.dropna(how="all", inplace=True)

    # Drop duplicate rows
    before = len(train)
    train.drop_duplicates(inplace=True)
    print(f"\nRemoved {before - len(train)} duplicate rows")

    # Drop CustomerID if exists
    if "CustomerID" in train.columns:
        train.drop(columns=["CustomerID"], inplace=True)
        print("Dropped CustomerID column")

    # Convert 'Last Interaction' to numeric days
    if "Last Interaction" in train.columns:
        train["Last Interaction"] = pd.to_datetime(train["Last Interaction"], errors="coerce")
        train["days_since_last_interaction"] = (pd.Timestamp.today() - train["Last Interaction"]).dt.days
        train.drop(columns=["Last Interaction"], inplace=True)
        print("Converted 'Last Interaction' to 'days_since_last_interaction'")

    # Handle categorical variables: Fill missing with mode
    cat_cols = train.select_dtypes(include=["object"]).columns
    for col in cat_cols:
        mode_val = train[col].mode()[0]
        train[col].fillna(mode_val, inplace=True)
        print(f"Filled missing values in '{col}' with mode: {mode_val}")

    # Handle numeric variables: Fill missing with median
    num_cols = train.select_dtypes(include=["number"]).columns
    for col in num_cols:
        median_val = train[col].median()
        train[col].fillna(median_val, inplace=True)
        print(f"Filled missing values in '{col}' with median: {median_val}")

    # Ensure target is clean
    if "Churn" in train.columns:
        missing_churn = train["Churn"].isna().sum()
        if missing_churn > 0:
            print(f"Dropping {missing_churn} rows with missing 'Churn'")
            train = train.dropna(subset=["Churn"])

    # Convert categorical to lowercase and strip spaces
    for col in cat_cols:
        train[col] = train[col].astype(str).str.strip().str.lower()

    # --- Outlier handling ---
    print("\nHandling outliers using IQR method...")
    for col in num_cols:
        q1 = train[col].quantile(0.25)
        q3 = train[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        outliers = ((train[col] < lower) | (train[col] > upper)).sum()
        if outliers > 0:
            train[col] = np.where(train[col] < lower, lower, train[col])
            train[col] = np.where(train[col] > upper, upper, train[col])
            print(f"Capped {outliers} outliers in '{col}'")

    # --- Correlation heatmap ---
    plt.figure(figsize=(10, 6))
    sns.heatmap(train[num_cols].corr(), annot=True, cmap="coolwarm")
    plt.title("Feature Correlation Heatmap")
    plt.tight_layout()
    plt.savefig("artifacts/correlation_heatmap.png")
    print("Saved correlation heatmap to artifacts/correlation_heatmap.png")

    # --- Save cleaned data ---
    out_train = Path("data/clean_train.csv")
    train.to_csv(out_train, index=False)
    print(f"\nCleaned training data saved to {out_train}")

    if test is not None:
        # Apply same cleaning to test dataset
        print("\nCleaning test dataset similarly...")
        if "CustomerID" in test.columns:
            test.drop(columns=["CustomerID"], inplace=True)
        if "Last Interaction" in test.columns:
            test["Last Interaction"] = pd.to_datetime(test["Last Interaction"], errors="coerce")
            test["days_since_last_interaction"] = (pd.Timestamp.today() - test["Last Interaction"]).dt.days
            test.drop(columns=["Last Interaction"], inplace=True)

        for col in cat_cols:
            if col in test.columns:
                mode_val = train[col].mode()[0]
                test[col].fillna(mode_val, inplace=True)
                test[col] = test[col].astype(str).str.strip().str.lower()
        for col in num_cols:
            if col in test.columns:
                median_val = train[col].median()
                test[col].fillna(median_val, inplace=True)

        out_test = Path("data/clean_test.csv")
        test.to_csv(out_test, index=False)
        print(f"Cleaned test data saved to {out_test}")

    print("\n✅ EDA and Cleaning Completed Successfully!")

if __name__ == "__main__":
    eda_and_clean("data/train.csv", "data/test.csv")
