import pandas as pd
import matplotlib.pyplot as plt
import os
import config
from data_loader import DataLoader
from processing import RankGaussProcessor
from src.trained_model import ChampionModel
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance

def run_test():
    print("========================================")
    print("   TEST 3: PERMUTATION IMPORTANCE       ")
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
    X_test_qt = processor.transform(X_test)
    
    # 3. Train Model
    print("Training model...")
    model = ChampionModel(config.XGB_PARAMS)
    model.train(X_train_qt, y_train)
    
    # 4. Run Permutation Check (The Honest Test)
    # We use the sklearn wrapper on the internal model
    print("Running Permutation (Scrambling features one by one)...")
    result = permutation_importance(
        model.model, X_test_qt, y_test, 
        n_repeats=5, random_state=42, n_jobs=-1, scoring='r2'
    )
    
    # 5. Process Results
    perm_sorted_idx = result.importances_mean.argsort()
    
    # 6. Plot
    plt.figure(figsize=(10, 8))
    plt.boxplot(
        result.importances[perm_sorted_idx].T,
        vert=False,
        labels=X_test_qt.columns[perm_sorted_idx]
    )
    plt.title('Test 3: Permutation Importance (Truth Check)')
    plt.xlabel('Drop in R² when feature is scrambled')
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    save_path = os.path.join('plots', 'permutation_importance.png')
    plt.savefig(save_path)
    print(f"\n[SUCCESS] Plot saved to {save_path}")
    print("Features at the bottom (near 0.0) are useless. Features with huge bars are vital.")

if __name__ == "__main__":
    run_test()