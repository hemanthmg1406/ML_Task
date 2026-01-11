import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
import os
import config
from data_loader import DataLoader
from processing import RankGaussProcessor
from src.trained_model import ChampionModel
from sklearn.model_selection import train_test_split

def run_test():
    print("========================================")
    print("   TEST 1: NATIVE FEATURE IMPORTANCE    ")
    print("========================================")
    
    # 1. Load Data
    loader = DataLoader(config.DATA_PATH, config.TARGET_PATH)
    X_raw, y_raw = loader.load()
    X_selected = loader.select_features(X_raw, config.TOP_30_FEATURES)
    
    # 2. Split & Process
    X_train, X_test, y_train, y_test = train_test_split(
        X_selected, y_raw, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE
    )
    
    processor = RankGaussProcessor(config.RANDOM_STATE)
    X_train_qt = processor.fit_transform(X_train)
    
    # 3. Train Model
    print("Training model to check feature weights...")
    model = ChampionModel(config.XGB_PARAMS)
    model.train(X_train_qt, y_train)
    
    # 4. Extract Importance
    # We access the internal booster to get 'total_gain'
    importance = model.model.get_booster().get_score(importance_type='total_gain')
    
    # Sort
    sorted_importance = sorted(importance.items(), key=lambda x: x[1], reverse=True)
    features, scores = zip(*sorted_importance)
    
    # 5. Plot
    plt.figure(figsize=(12, 8))
    plt.barh(features[:30], scores[:30], color='skyblue')
    plt.gca().invert_yaxis() # Highest at top
    plt.title('Test 1: XGBoost Feature Importance (Total Gain)')
    plt.xlabel('Total Gain (Contribution to Accuracy)')
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    
    os.makedirs('plots', exist_ok=True)
    save_path = os.path.join('plots', 'feature_importance_gain.png')
    plt.savefig(save_path)
    print(f"\n[SUCCESS] Plot saved to {save_path}")
    print("Check this plot. If the bottom features are near zero, we should remove them.")

if __name__ == "__main__":
    run_test()