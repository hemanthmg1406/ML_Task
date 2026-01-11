import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
import os
import config
import numpy as np
from data_loader import DataLoader
from processing import RankGaussProcessor
from src.trained_model import ChampionModel
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

def run_test():
    print("========================================")
    print("   TEST 2: THE ELBOW TEST (QUANTITY)    ")
    print("========================================")

    # 1. Load ALL Data (Ignore config.TOP_30 for a moment)
    loader = DataLoader(config.DATA_PATH, config.TARGET_PATH)
    X_raw, y_raw = loader.load()
    
    # 2. Rank ALL Features first
    # We train a quick model on everything to find the "Truth" ranking
    print("Ranking ALL available features...")
    processor = RankGaussProcessor(config.RANDOM_STATE)
    X_qt = processor.fit_transform(X_raw)
    
    temp_model = xgb.XGBRegressor(n_estimators=100, n_jobs=-1, random_state=42)
    temp_model.fit(X_qt, y_raw)
    
    # Get sorted list of ALL features
    imp = temp_model.get_booster().get_score(importance_type='total_gain')
    sorted_features = [k for k, v in sorted(imp.items(), key=lambda x: x[1], reverse=True)]
    
    print(f"Total available features: {len(sorted_features)}")
    
    # 3. The Loop
    steps = [10, 20, 30, 40, 50, 60, len(sorted_features)]
    steps = sorted(list(set([s for s in steps if s <= len(sorted_features)])))
    
    r2_scores = []
    
    for k in steps:
        print(f"\nTesting Top {k} features...")
        top_k = sorted_features[:k]
        
        # Select Data
        X_selected = X_raw[top_k]
        
        # Split
        X_train, X_test, y_train, y_test = train_test_split(
            X_selected, y_raw, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE
        )
        
        # Process
        proc = RankGaussProcessor(config.RANDOM_STATE)
        X_train_q = proc.fit_transform(X_train)
        X_test_q = proc.transform(X_test)
        
        # Train Robust Model
        model = ChampionModel(config.XGB_PARAMS)
        model.train(X_train_q, y_train)
        
        # Evaluate
        preds = model.predict(X_test_q)
        score = r2_score(y_test, preds)
        r2_scores.append(score)
        print(f"   -> R²: {score:.4f}")

    # 4. Plot
    plt.figure(figsize=(10, 6))
    plt.plot(steps, r2_scores, marker='o', linewidth=2, color='green')
    plt.title('Test 2: The Elbow Curve (Quantity vs Accuracy)')
    plt.xlabel('Number of Features')
    plt.ylabel('Test R² Score')
    plt.grid(True)
    plt.axvline(x=30, color='red', linestyle='--', label='Current Selection (30)')
    plt.legend()
    
    save_path = os.path.join('plots', 'feature_quantity_elbow.png')
    plt.savefig(save_path)
    print(f"\n[SUCCESS] Plot saved to {save_path}")
    print("Look for the 'Elbow'. If the line is still going UP after 30, we need more features!")

if __name__ == "__main__":
    run_test()