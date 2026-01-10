import pandas as pd
import numpy as np
import joblib  # <--- CHANGED: Using joblib instead of pickle
import xgboost as xgb
import os

# --- CONFIGURATION ---
MODEL_PATH = 'artifacts/champion_model.json'
PREPROCESSOR_PATH = 'artifacts/preprocessor.pkl'
DATA_PATH = 'data/dataset_30.csv'
TARGET_PATH = 'data/target_30.csv'

def run_diagnostics():
    print("========================================")
    print("   PIPELINE DIAGNOSTICS (joblib fix)    ")
    print("========================================")

    # 1. LOAD ARTIFACTS
    # ------------------------------------------------
    if not os.path.exists(MODEL_PATH) or not os.path.exists(PREPROCESSOR_PATH):
        print(f"ERROR: Artifacts not found.")
        return

    print(f"[1] Inspecting Model Artifact: {MODEL_PATH}")
    model = xgb.Booster()
    model.load_model(MODEL_PATH)
    
    # Extract feature names stored inside the XGBoost model
    model_features = model.feature_names
    if model_features is None:
        print("    ! Warning: Model does not have saved feature names.")
        model_n_features = model.num_features()
    else:
        model_n_features = len(model_features)
    
    print(f"    -> The Trained Model expects exactly {model_n_features} features.")
    if model_features:
        print(f"    -> First 5 Expected Features: {model_features[:5]}")

    print(f"\n[2] Inspecting Preprocessor: {PREPROCESSOR_PATH}")
    try:
        # CHANGED: Using joblib.load
        preprocessor = joblib.load(PREPROCESSOR_PATH)
        print(f"    -> Successfully loaded using joblib.")
        print(f"    -> Type: {type(preprocessor)}")
    except Exception as e:
        print(f"    [CRITICAL FAIL] Could not load preprocessor: {e}")
        return

    # 2. LOAD DATA
    # ------------------------------------------------
    print(f"\n[3] Inspecting New Data: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    print(f"    -> Loaded Shape: {df.shape}")

    # 3. CRITICAL CHECK: ALIGNMENT
    # ------------------------------------------------
    print("\n========================================")
    print("   DIAGNOSIS 1: FEATURE ALIGNMENT")
    print("========================================")
    
    if model_features:
        # Check if the CSV columns match the Model features
        missing_cols = [f for f in model_features if f not in df.columns]
        
        if len(missing_cols) > 0:
            print(f" [CRITICAL FAIL] The CSV is missing {len(missing_cols)} features expected by the model!")
            print(f"    -> Examples missing: {missing_cols[:3]}")
        else:
            print(" [PASS] All model features exist in the CSV.")
            
            # Check Order
            match = (list(df.columns[:5]) == model_features[:5])
            if not match:
                print(" [WARNING] Column Order Mismatch!")
                print(f"    -> Model expects: {model_features[:5]}")
                print(f"    -> CSV provides:  {list(df.columns[:5])}")
                print("    -> CONCLUSION: The features are shuffled. This causes negative R2.")
            else:
                print(" [PASS] Column order appears aligned.")

    # 4. CRITICAL CHECK: PREDICTION SCALE
    # ------------------------------------------------
    print("\n========================================")
    print("   DIAGNOSIS 2: TARGET SCALE (Log vs Raw)")
    print("========================================")
    
    try:
        # Prepare Input: strictly select the columns the model wants
        if model_features:
            # We filter the dataframe to only the columns the model knows
            # If columns are missing, this will fail (caught by try/except)
            input_data = df[model_features].head(5)
        else:
            input_data = df.iloc[:, :model_n_features].head(5)

        # Apply preprocessor
        # We assume standard sklearn API (transform)
        try:
            print("    -> Transforming data...")
            processed_data = preprocessor.transform(input_data)
            dmatrix = xgb.DMatrix(processed_data, feature_names=model_features)
        except Exception as e:
            print(f" ! Transform failed: {e}")
            print(" ! Attempting raw prediction (ignoring scaler)...")
            dmatrix = xgb.DMatrix(input_data, feature_names=model_features)

        preds = model.predict(dmatrix)
        
        # Load Truth
        truth = pd.read_csv(TARGET_PATH).head(5)
        truth_values = truth.iloc[:, 0].values

        print(f"\n{'Prediction':<15} | {'Ground Truth':<15} | {'Diff'}")
        print("-" * 45)
        for p, t in zip(preds, truth_values):
            print(f"{p:<15.4f} | {t:<15.4f} | {p-t:.4f}")

        # Check for Log Scale
        avg_pred = np.mean(preds)
        avg_true = np.mean(truth_values)
        
        print("\n--- ANALYSIS ---")
        if avg_true > (avg_pred * 5):
            print(" [CONCLUSION] HUGE SCALE MISMATCH.")
            print("    -> Model predicts small values (likely Log/Normalized).")
            print("    -> Truth is large values (Raw).")
        else:
             print(" [PASS] Scale seems correct.")
            
    except Exception as e:
        print(f"Could not generate predictions: {e}")

if __name__ == "__main__":
    run_diagnostics()