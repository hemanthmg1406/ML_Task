import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
import os
import config
from data_loader import DataLoader
from processing import RankGaussProcessor

def run_null_importance():
    print("========================================")
    print("   TEST 5: NULL IMPORTANCE (The Sham Check) ")
    print("========================================")
    
    # 1. Load Data
    loader = DataLoader(config.DATA_PATH, config.TARGET_PATH)
    X_raw, y_raw = loader.load()
    X_selected = loader.select_features(X_raw, config.TOP_10_FEATURES)
    
    # 2. Process
    processor = RankGaussProcessor(config.RANDOM_STATE)
    X_qt = processor.fit_transform(X_selected)
    
    # 3. Calculate Actual Importance
    print("Calculating True Importance...")
    model = xgb.XGBRegressor(**config.XGB_PARAMS)
    model.fit(X_qt, y_raw)
    real_imp = model.feature_importances_
    
    # 4. Calculate Null Importances (50 Shuffles)
    print("Running 50 Null Shuffles (This checks if features are just lucky)...")
    null_imps = []
    n_runs = 50
    
    for i in range(n_runs):
        # Shuffle targets to break real relationship
        y_shuff = np.random.permutation(y_raw)
        
        # Train "Fake" Model
        null_model = xgb.XGBRegressor(**config.XGB_PARAMS)
        null_model.fit(X_qt, y_shuff)
        
        null_imps.append(null_model.feature_importances_)
        
        if (i+1) % 10 == 0:
            print(f"   - Finished shuffle {i+1}/{n_runs}")
            
    null_imps = np.array(null_imps) # Shape: (50, 10)
    
    # 5. Plot Comparison
    print("\nGenerating Plots...")
    features = config.TOP_30_FEATURES
    n_feats = len(features)
    
    fig, axes = plt.subplots(n_feats, 1, figsize=(8, 20))
    if n_feats == 1: axes = [axes]
    
    for i, ax in enumerate(axes):
        feature_name = features[i]
        
        # Plot Null Distribution (The Noise)
        ax.hist(null_imps[:, i], bins=10, alpha=0.5, color='gray', label='Null (Random Chance)')
        
        # Plot True Importance (The Signal)
        ax.axvline(x=real_imp[i], color='red', linewidth=3, label='True Importance')
        
        ax.set_title(f"Feature: {feature_name}")
        ax.legend()
        
    plt.tight_layout()
    os.makedirs('plots', exist_ok=True)
    save_path = os.path.join('plots', 'null_importance_test.png')
    plt.savefig(save_path)
    print(f"\n[SUCCESS] Saved {save_path}")
    print("Interpreting the plot: The RED LINE should be far to the right of the GRAY BARS.")
    print("If the red line touches the gray bars, that feature is fake.")

if __name__ == "__main__":
    run_null_importance()