import pandas as pd
import matplotlib.pyplot as plt
import config
import os
from data_loader import DataLoader
from processing import RankGaussProcessor
from src.trained_model import ChampionModel
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance

def run_final_permutation():
    print("========================================")
    print("   TEST 4: PERMUTATION (Train vs Test)  ")
    print("========================================")
    
    # 1. Load & Split
    loader = DataLoader(config.DATA_PATH, config.TARGET_PATH)
    X_raw, y_raw = loader.load()
    X_selected = loader.select_features(X_raw, config.TOP_30_FEATURES)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_selected, y_raw, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE
    )
    
    # 2. Process
    proc = RankGaussProcessor(config.RANDOM_STATE)
    X_train_q = proc.fit_transform(X_train)
    X_test_q = proc.transform(X_test)
    
    # 3. Train
    print("Training Final Model...")
    model = ChampionModel(config.XGB_PARAMS)
    model.train(X_train_q, y_train)
    
    # 4. Compute Permutation Importance
    print("Computing Importance on TRAIN set (Memorization Check)...")
    r_train = permutation_importance(model.model, X_train_q, y_train, n_repeats=10, random_state=42, n_jobs=-1)
    
    print("Computing Importance on TEST set (Generalization Check)...")
    r_test = permutation_importance(model.model, X_test_q, y_test, n_repeats=10, random_state=42, n_jobs=-1)
    
    # 5. Plot
    sorted_idx = r_test.importances_mean.argsort()
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot Test (Green)
    ax.boxplot(
        r_test.importances[sorted_idx].T,
        vert=False,
        patch_artist=True,
        boxprops=dict(facecolor="lightgreen", color="green"),
        labels=X_test_q.columns[sorted_idx]
    )
    
    # Plot Train (Red outline) - to compare gap
    # We overlay the means just to see the gap
    ax.scatter(r_train.importances_mean[sorted_idx], range(len(sorted_idx)), color='red', label='Train Mean', zorder=5)
    
    ax.set_title("Permutation Importance: Test (Green Box) vs Train (Red Dot)")
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.5)
    
    save_path = os.path.join('plots', 'permutation_final.png')
    plt.savefig(save_path)
    print(f"\n[SUCCESS] Saved {save_path}")
    print("Look for gaps: If the Red Dot is far to the right of the Green Box, the model is overfitting that feature.")

if __name__ == "__main__":
    run_final_permutation()