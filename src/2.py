import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import xgboost as xgb
import os
import config
from data_loader import DataLoader
from processing import RankGaussProcessor

def finalize_features():
    print("========================================")
    print("   STAGE 2: HYGIENE CHECK (TOP 10)      ")
    print("========================================")

    # 1. Load & Rank to get the Golden 10
    loader = DataLoader(config.DATA_PATH, config.TARGET_PATH)
    X_raw, y_raw = loader.load()
    
    processor = RankGaussProcessor(config.RANDOM_STATE)
    X_qt = processor.fit_transform(X_raw)
    
    print("Re-ranking to identify the Golden 10...")
    model = xgb.XGBRegressor(n_estimators=100, n_jobs=-1, random_state=42)
    model.fit(X_qt, y_raw)
    
    imp = model.get_booster().get_score(importance_type='total_gain')
    sorted_features = [k for k, v in sorted(imp.items(), key=lambda x: x[1], reverse=True)]
    
    golden_10 = sorted_features[:10]
    
    print("\n" + "="*40)
    print("THE GOLDEN 10 LIST (For config.py)")
    print("="*40)
    print(golden_10)
    print("="*40)

    # 2. Correlation Check (Hygiene)
    print("\nChecking for Redundancy (Correlation > 0.95)...")
    X_golden = X_raw[golden_10]
    corr_matrix = X_golden.corr().abs()
    
    # Filter for high correlation
    high_corr_pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if corr_matrix.iloc[i, j] > 0.95:
                pair = (corr_matrix.columns[i], corr_matrix.columns[j], corr_matrix.iloc[i, j])
                high_corr_pairs.append(pair)
    
    if high_corr_pairs:
        print(f"[WARNING] Found {len(high_corr_pairs)} highly correlated pairs!")
        for p in high_corr_pairs:
            print(f"   - {p[0]} vs {p[1]}: {p[2]:.4f}")
        print("-> ACTION: We should drop one of these and take Feature #11 instead.")
    else:
        print("[PASS] No redundancy found. These 10 features are unique.")

    # 3. Save Correlation Plot
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', vmin=0, vmax=1)
    plt.title('Correlation of Golden 10 Features')
    
    os.makedirs('plots', exist_ok=True)
    save_path = os.path.join('plots', 'golden_10_correlation.png')
    plt.savefig(save_path)
    print(f"\n[INFO] Correlation plot saved to {save_path}")

if __name__ == "__main__":
    finalize_features()