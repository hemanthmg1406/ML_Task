"""
EDA for Part A: target01 regression

Purpose:
- Understand data integrity
- Analyze target01 behavior
- Inspect feature scales
- Explore feature–target relationships

IMPORTANT:
- Read-only analysis
- No data modification
- No CSV writing
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# --------------------------------------------------
# EDA-0: Setup & scope confirmation
# --------------------------------------------------


def load_data():
    X = pd.read_csv("data/dataset_29.csv")
    y = pd.read_csv("data/target_29.csv")
    return X, y


def check_basic_shapes(X, y):
    print("Feature matrix shape:", X.shape)
    print("Target shape:", y.shape)

    if X.shape[0] != y.shape[0]:
        raise ValueError("Row count mismatch between X and y")

    print("All feature dtypes:")
    print(X.dtypes.value_counts())

    print("Target columns:", y.columns.tolist())



# --------------------------------------------------
# EDA-1: Data integrity & cleanliness
# --------------------------------------------------

def check_missing_and_constants(X):
    print("\n--- Missing values check ---")
    missing_counts = X.isna().sum()
    total_missing = missing_counts.sum()
    print("Total missing values:", total_missing)

    if total_missing > 0:
        print("Columns with missing values:")
        print(missing_counts[missing_counts > 0])

    print("\n--- Duplicate rows check ---")
    duplicate_rows = X.duplicated().sum()
    print("Number of duplicate rows:", duplicate_rows)

    print("\n--- Constant / near-constant features ---")
    nunique = X.nunique()
    constant_features = nunique[nunique <= 1]

    print("Number of constant features:", constant_features.shape[0])
    if constant_features.shape[0] > 0:
        print("Constant feature names:")
        print(constant_features.index.tolist())

def basic_descriptive_stats(X):
    # TODO:
    # - compute mean, std, min, max
    pass


# --------------------------------------------------
# EDA-2: Target (target01) analysis
# --------------------------------------------------

def analyze_target(y):
    print("\n--- Target01 analysis ---")

    target = y["target01"]

    print("Summary statistics:")
    print(target.describe())

    print("\nSkewness of target01:")
    print(target.skew())

    plt.figure(figsize=(6, 4))
    plt.hist(target, bins=50)
    plt.title("Distribution of target01")
    plt.xlabel("target01")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.show()
    plt.close()

# --------------------------------------------------
# EDA-3: Feature scale & distribution inspection
# --------------------------------------------------

def inspect_feature_scales(X):
    print("\n>>> ENTERED EDA-3 <<<")
    print("\n--- Feature scale inspection ---")
    stds = X.std()
    print("Feature standard deviation summary:")
    print(stds.describe())
    smallest_std = stds.nsmallest(5)
    largest_std = stds.nlargest(5)
    print("\nFeatures with smallest standard deviation:")
    print(smallest_std)
    print("\nFeatures with largest standard deviation:")
    print(largest_std)



def plot_selected_feature_distributions(X):
    selected_features = [
        X.columns[0],
        X.columns[len(X.columns) // 2],
        X.columns[-1],
    ]

    plt.figure(figsize=(10, 3))
    for i, col in enumerate(selected_features, 1):
        plt.subplot(1, 3, i)
        plt.hist(X[col], bins=50)
        plt.title(col)
    plt.tight_layout()
    plt.show()
    plt.close()

# --------------------------------------------------
# EDA-4: Feature–target relationship
# --------------------------------------------------

def feature_target_correlation(X, y):
    print("\n--- Feature–target correlation (target01) ---")

    target = y["target01"]
    correlations = X.corrwith(target)

    correlations_sorted = correlations.abs().sort_values(ascending=False)

    print("Top 15 features by absolute correlation with target01:")
    print(correlations_sorted.head(15))



# --------------------------------------------------
# EDA-5: EDA conclusions (written, not computed)
# --------------------------------------------------

def summarize_findings():
    # TODO:
    # - scaling needed?
    # - target behavior notes
    # - signal strength impressions
    pass


# --------------------------------------------------
# Main execution (EDA only)
# --------------------------------------------------

def main():
    X, y = load_data()
    check_basic_shapes(X, y)
    check_missing_and_constants(X)
    analyze_target(y)
    inspect_feature_scales(X)
    plot_selected_feature_distributions(X)
    feature_target_correlation(X, y)
    #summarize_findings()'''


if __name__ == "__main__":
    main()
