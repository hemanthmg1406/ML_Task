import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, KFold
from xgboost import XGBRegressor  # <--- CHANGED TO REGRESSOR

# ==========================================
# 1. SETUP & DATA LOADING
# ==========================================

print("Loading data...")

try:
    # Load Features
    X = pd.read_csv("./data/dataset_29.csv")
    
    # Load Target
    y = pd.read_csv("./data/target_29.csv")
    
    # Ensure y is a Series and KEEP AS FLOAT (Continuous)
    if isinstance(y, pd.DataFrame):
        y = y.iloc[:, 0]

    print("Data loaded successfully.")
    print(f"Features Shape: {X.shape}")
    print(f"Target Shape:   {y.shape}")
    print(f"Target Type:    {y.dtype}") # Should be float64

except Exception as e:
    print(f"\n[ERROR] Could not load data.\nDetails: {e}")
    exit()

# ==========================================
# 2. DEFINE FEATURE SETS (From your Plot)
# ==========================================

noise_features = [
    'feat_124', 'feat_165', 'feat_266', 'feat_101', 
    'feat_200', 'feat_112', 'feat_207'
]

all_features = X.columns.tolist()
clean_features = [f for f in all_features if f not in noise_features]

print(f"\nOriginal Feature Count: {len(all_features)}")
print(f"Filtered Feature Count: {len(clean_features)}")

# ==========================================
# 3. VERIFICATION: The "Ablation Test" (Regression)
# ==========================================

def verify_feature_drop(X, y, features_all, features_clean):
    """
    Compares model performance with and without noise features 
    using K-Fold for Regression.
    """
    # XGBoost Regressor configuration
    model = XGBRegressor(
        n_estimators=100, 
        max_depth=4, 
        learning_rate=0.05, 
        random_state=42,
        n_jobs=-1,
        objective='reg:squarederror' # Standard regression objective
    )
    
    # KFold is standard for regression (Stratified is impossible for continuous targets)
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    
    print("\n--- Starting Verification (Metric: RMSE) ---")
    print("Note: Values will be negative (sklearn convention). Closer to 0 is better.")
    
    # 1. Test Baseline (All Features)
    print("Testing Baseline (All Features)...")
    # We use neg_root_mean_squared_error. 
    # E.g., -0.15 is BETTER than -0.20
    scores_base = cross_val_score(model, X[features_all], y, cv=cv, scoring='neg_root_mean_squared_error')
    avg_base = np.mean(scores_base)
    std_base = np.std(scores_base)
    print(f"Baseline RMSE: {avg_base:.5f} (+/- {std_base:.5f})")
    
    # 2. Test Candidate (Filtered Features)
    print("Testing Cleaned (Hard Filter)...")
    scores_clean = cross_val_score(model, X[features_clean], y, cv=cv, scoring='neg_root_mean_squared_error')
    avg_clean = np.mean(scores_clean)
    std_clean = np.std(scores_clean)
    print(f"Cleaned  RMSE: {avg_clean:.5f} (+/- {std_clean:.5f})")
    
    # 3. Conclusion
    # Since scores are negative (e.g. -0.5), a positive difference means improvement.
    diff = avg_clean - avg_base
    
    print("\n--- Verdict ---")
    # We accept if the clean model is better (higher score) or basically the same (diff > -0.001)
    if diff >= -0.001: 
        print("✅ SUCCESS: Removing features maintained performance.")
        print(f"   Improvement: {diff:.5f} (Higher is better)")
        print("   Action: PROCEED with 'clean_features' only.")
    else:
        print("⚠️ WARNING: Performance dropped.")
        print(f"   Loss: {diff:.5f}")
        print("   Action: Re-evaluate 'noise_features'.")

# Run the verification
verify_feature_drop(X, y, all_features, clean_features)