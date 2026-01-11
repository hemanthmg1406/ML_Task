import pandas as pd
import xgboost as xgb
import config
from data_loader import DataLoader
from processing import RankGaussProcessor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

def run_micro_elbow():
    print("========================================")
    print("   MICRO-ELBOW TEST (10 to 15)          ")
    print("========================================")

    # 1. Load & Rank
    loader = DataLoader(config.DATA_PATH, config.TARGET_PATH)
    X_raw, y_raw = loader.load()
    
    processor = RankGaussProcessor(config.RANDOM_STATE)
    X_qt = processor.fit_transform(X_raw)
    
    print("Ranking features...")
    model = xgb.XGBRegressor(n_estimators=100, n_jobs=-1, random_state=42)
    model.fit(X_qt, y_raw)
    
    imp = model.get_booster().get_score(importance_type='total_gain')
    sorted_features = [k for k, v in sorted(imp.items(), key=lambda x: x[1], reverse=True)]
    
    # 2. The Micro-Loop
    feature_counts = [10, 11, 12, 13, 14, 15]
    results = {}

    print("\nStarting comparison...")
    print(f"{'Count':<6} | {'R² Score':<10} | {'Added Feature Name'}")
    print("-" * 45)

    for k in feature_counts:
        # Pick Top K
        current_feat = sorted_features[:k]
        
        # Identify what was added (the new feature vs previous)
        new_feat = current_feat[-1] 
        
        # Train/Test
        X_selected = X_raw[current_feat]
        X_train, X_test, y_train, y_test = train_test_split(
            X_selected, y_raw, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE
        )
        
        proc = RankGaussProcessor(config.RANDOM_STATE)
        X_train_q = proc.fit_transform(X_train)
        X_test_q = proc.transform(X_test)
        
        champion = xgb.XGBRegressor(**config.XGB_PARAMS)
        champion.fit(X_train_q, y_train)
        
        score = r2_score(y_test, champion.predict(X_test_q))
        results[k] = score
        
        print(f"{k:<6} | {score:.5f}    | {new_feat}")

    # 3. Recommendation
    best_k = max(results, key=results.get)
    print("-" * 45)
    print(f"[RESULT] The absolute peak is at {best_k} features (R²: {results[best_k]:.5f})")

if __name__ == "__main__":
    run_micro_elbow()