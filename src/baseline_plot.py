import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.inspection import permutation_importance
from xgboost import XGBRegressor
import os

# -----------------------
# CONFIG
# -----------------------
DATA_PATH = "data/dataset_29.csv"
PLOTS_DIR = "plots"

os.makedirs(PLOTS_DIR, exist_ok=True)

# -----------------------
# Load raw data
# -----------------------
df = pd.read_csv(DATA_PATH)

# column 1 is target (0-indexed)
y = df.iloc[:, 1]
X = df.drop(df.columns[1], axis=1)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------
# Baseline raw XGBoost
# -----------------------
model = XGBRegressor(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=1.0,
    colsample_bytree=1.0,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

# -----------------------
# Noise & drift
# -----------------------
def add_noise(X, sigma=0.05):
    return X + np.random.normal(0, sigma, X.shape)

def add_drift(X, scale=0.15):
    return X * (1 + scale)

X_test_clean = X_test.copy()
X_test_noise = add_noise(X_test)
X_test_drift = add_drift(X_test)

# -----------------------
# Predictions
# -----------------------
y_pred_clean = model.predict(X_test_clean)
y_pred_noise = model.predict(X_test_noise)
y_pred_drift = model.predict(X_test_drift)

# -----------------------
# R² scores
# -----------------------
r2_clean = r2_score(y_test, y_pred_clean)
r2_noise = r2_score(y_test, y_pred_noise)
r2_drift = r2_score(y_test, y_pred_drift)

print("\nBaseline raw XGBoost")
print("Clean R2 :", r2_clean)
print("Noise R2 :", r2_noise)
print("Drift R2 :", r2_drift)

# -----------------------
# Plot 1 – Robustness
# -----------------------
plt.figure(figsize=(6,4))
plt.bar(["Clean","Noise","Drift"], [r2_clean, r2_noise, r2_drift])
plt.ylabel("R²")
plt.title("Baseline Raw XGBoost – Robustness")
plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/baseline_raw_robustness.png", dpi=150)
plt.close()

# -----------------------
# Plot 2 – Prediction instability
# -----------------------
plt.figure(figsize=(5,5))
plt.scatter(y_pred_clean, y_pred_noise, alpha=0.5)
plt.xlabel("Clean Predictions")
plt.ylabel("Noisy Predictions")
plt.title("Prediction Shift (Raw Baseline)")
plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/baseline_raw_prediction_shift.png", dpi=150)
plt.close()

# -----------------------
# Plot 3 – Feature instability
# -----------------------
r_clean = permutation_importance(model, X_test_clean, y_test, n_repeats=10, random_state=42)
r_noise = permutation_importance(model, X_test_noise, y_test, n_repeats=10, random_state=42)

std_shift = np.abs(r_clean.importances_mean - r_noise.importances_mean)

top_idx = np.argsort(std_shift)[-30:]
features = X.columns[top_idx]
values = std_shift[top_idx]

plt.figure(figsize=(10,4))
plt.bar(range(len(features)), values)
plt.xticks(range(len(features)), features, rotation=90)
plt.ylabel("Importance shift |clean − noise|")
plt.title("Feature Instability – Raw Baseline")
plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/baseline_raw_feature_instability.png", dpi=150)
plt.close()

print("\nSaved plots to /plots")
