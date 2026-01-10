import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
from sklearn.metrics import r2_score

# --- CONFIGURATION ---
DATA_PATH = 'data/dataset_30.csv'
TARGET_PATH = 'data/target_30.csv'
MODEL_PATH = 'artifacts/champion_model.json'
PREPROCESSOR_PATH = 'artifacts/preprocessor.pkl'

def evaluate_double_align():
    print("==================================================")
    print("   PRODUCTION EVALUATION: DOUBLE ALIGNMENT CHECK  ")
    print("==================================================")

    # 1. LOAD ARTIFACTS
    # ------------------------------------------------
    print("\n[Step 1] Loading Artifacts...")
    model = xgb.Booster()
    model.load_model(MODEL_PATH)
    model_features = model.feature_names
    print(f" - Model expects {len(model_features)} features.")
    print(f" - Model First 3: {model_features[:3]}")

    preprocessor = joblib.load(PREPROCESSOR_PATH)
    print(f" - Preprocessor loaded: {type(preprocessor)}")

    # 2. CHECK PREPROCESSOR MEMORY
    # ------------------------------------------------
    # We need to know what order the Preprocessor learned.
    if hasattr(preprocessor, 'feature_names_in_'):
        prep_features = list(preprocessor.feature_names_in_)
        print(f" - Preprocessor expects {len(prep_features)} features.")
        print(f" - Preprocessor First 3: {prep_features[:3]}")
        
        # DEBUG: Check if orders are different
        if prep_features == model_features:
            print(" [INFO] Preprocessor and Model want the EXACT SAME order.")
        else:
            print(" [CRITICAL DISCOVERY] ORDERS ARE DIFFERENT!")
            print(" -> This confirms the mismatch theory.")
            print(f" -> Prep wants: {prep_features[:3]}")
            print(f" -> Modl wants: {model_features[:3]}")
    else:
        print(" [ERROR] Preprocessor does not have 'feature_names_in_'.")
        print(" -> Cannot safely reorder inputs for the preprocessor.")
        print(" -> Proceeding with Model Order (High Risk)...")
        prep_features = model_features # Fallback

    # 3. LOAD DATA
    # ------------------------------------------------
    print(f"\n[Step 2] Loading Data: {DATA_PATH}...")
    df = pd.read_csv(DATA_PATH)

    # 4. ALIGNMENT 1: FOR PREPROCESSOR
    # ------------------------------------------------
    print("\n[Step 3] Alignment 1: Feeding the Preprocessor...")
    try:
        # Force data to match Preprocessor's expectation
        X_prep = df[prep_features]
        print(" - Data reordered to match Preprocessor.")
        
        # Transform
        X_trans_array = preprocessor.transform(X_prep)
        
        # Convert back to DataFrame to preserve names for the next step
        X_trans_df = pd.DataFrame(X_trans_array, columns=prep_features)
        print(" - Transformation complete.")
        
    except KeyError as e:
        print(f" [FAIL] Missing columns for preprocessor: {e}")
        return

    # 5. ALIGNMENT 2: FOR MODEL
    # ------------------------------------------------
    print("\n[Step 4] Alignment 2: Feeding the Model...")
    try:
        # Now shuffle the TRANSFORMED data to match the MODEL'S expectation
        X_final = X_trans_df[model_features]
        print(" - Transformed data re-sorted to match Model.")
        
        # Create DMatrix
        dtest = xgb.DMatrix(X_final, feature_names=model_features)
        
    except KeyError as e:
        print(f" [FAIL] Missing columns for model re-sort: {e}")
        return

    # 6. PREDICT & EVALUATE
    # ------------------------------------------------
    print("\n[Step 5] Prediction & Evaluation...")
    preds = model.predict(dtest)
    
    target = pd.read_csv(TARGET_PATH)
    y_true = target.iloc[:, 0].values
    
    r2 = r2_score(y_true, preds)
    corr = np.corrcoef(preds, y_true)[0, 1]

    print("\n========================================")
    print("FINAL RESULTS")
    print("========================================")
    print(f"R2 SCORE:    {r2:.5f}")
    print(f"CORRELATION: {corr:.4f}")
    print("========================================")

if __name__ == "__main__":
    evaluate_double_align()