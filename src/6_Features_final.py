import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
from sklearn.inspection import permutation_importance
import processing
import config
import json
import os

def run_stable_feature_selection():
    print("--- STARTING BOOTSTRAPPED FEATURE SELECTION ---")
    # Load raw data
    X, y = processing.load_data(config.DATA_PATH, config.TARGET_PATH)
    
    # Initialize Transformers
    feat_proc = processing.RankGaussProcessor(config.RANDOM_STATE)
    target_proc = processing.TargetTransformer()
    
    # 5-Fold Cross Validation for Stability
    kf = KFold(n_splits=5, shuffle=True, random_state=config.RANDOM_STATE)
    importance_list = []

    # Use a conservative model for selection to avoid picking up on overfitted noise
    selector_model = xgb.XGBRegressor(
        n_estimators=500,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=config.RANDOM_STATE,
        n_jobs=-1
    )

    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        print(f"Processing Fold {fold+1}...")
        
        # Split data
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        # Transform Features and Target
        X_train_t = feat_proc.fit_transform(X_train)
        X_val_t = feat_proc.transform(X_val)
        y_train_t = target_proc.fit_transform(y_train)
        y_val_t = target_proc.transform(y_val)
        
        # Fit Model
        selector_model.fit(X_train_t, y_train_t)
        
        # Permutation Importance (Measures R2 drop when columns are shuffled)
        result = permutation_importance(
            selector_model, X_val_t, y_val_t, 
            n_repeats=3, random_state=config.RANDOM_STATE, n_jobs=-1
        )
        
        importance_list.append(result.importances_mean)

    # Aggregate Results
    mean_importance = np.mean(importance_list, axis=0)
    std_importance = np.std(importance_list, axis=0)
    
    # Stability Score: Mean Importance / Standard Deviation
    # Higher means the feature is consistently important across all folds
    stability_score = mean_importance / (std_importance + 1e-9)
    
    feat_stats = pd.DataFrame({
        'feature': X.columns,
        'mean_importance': mean_importance,
        'stability': stability_score
    }).sort_values(by='mean_importance', ascending=False)

    # Extract Top 100
    top_100 = feat_stats.head(100)['feature'].tolist()
    
    print("\nTop 10 Stable Features discovered:")
    print(feat_stats.head(10))

    # Save artifact for copy-pasting or use in config.py
    os.makedirs(config.ARTIFACT_DIR, exist_ok=True)
    output_path = os.path.join(config.ARTIFACT_DIR, 'top_100_features.json')
    with open(output_path, 'w') as f:
        json.dump(top_100, f)
    
    print(f"\nSuccess! Top 100 features saved to {output_path}")

if __name__ == "__main__":
    run_stable_feature_selection()