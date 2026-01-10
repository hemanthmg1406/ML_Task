import pandas as pd
import joblib
import xgboost as xgb
import os
import config

def generate_submission():
    print("   GENERATING FINAL SUBMISSION FILE     ")

    # 1. Define Paths
    EVAL_PATH = os.path.join('data', 'EVAL_29.csv')
    OUTPUT_NAME = 'EVAL_target01_29.csv'
    MODEL_PATH = os.path.join(config.ARTIFACT_DIR, 'champion_model.json')
    PREPROC_PATH = os.path.join(config.ARTIFACT_DIR, 'preprocessor.pkl')

    # 2. Load Artifacts (The Frozen Pipeline)
    print(f"[1/5] Loading artifacts from {config.ARTIFACT_DIR}...")
    try:
        model = xgb.XGBRegressor()
        model.load_model(MODEL_PATH)
        preprocessor = joblib.load(PREPROC_PATH)
        print("   - Model and Preprocessor loaded.")
    except Exception as e:
        print(f"[CRITICAL ERROR] Artifacts missing: {e}")
        print("Run 'python main.py' first!")
        return

    # 3. Load Evaluation Data
    print(f"[2/5] Loading Evaluation Data: {EVAL_PATH}...")
    try:
        X_eval_raw = pd.read_csv(EVAL_PATH)
        print(f"   - Loaded {len(X_eval_raw)} rows.")
    except FileNotFoundError:
        print(f"[CRITICAL ERROR] File not found: {EVAL_PATH}")
        return

    # 4. Feature Selection (Crucial Step)
    print(f"[3/5] Selecting the Champion Top {len(config.TOP_30_FEATURES)} features...")
    # Strict check: Ensure all top 30 features exist in EVAL
    missing_cols = [c for c in config.TOP_30_FEATURES if c not in X_eval_raw.columns]
    if missing_cols:
        print(f"[CRITICAL ERROR] EVAL file is missing columns: {missing_cols}")
        return
    
    X_eval_selected = X_eval_raw[config.TOP_30_FEATURES]

    # 5. Transform (Apply Frozen Rules)
    print("[4/5] Applying RankGauss Transform (Using Saved Rules)...")
    # NOTE: We use .transform(), NOT .fit_transform()
    X_eval_qt = pd.DataFrame(
        preprocessor.transform(X_eval_selected), 
        columns=config.TOP_30_FEATURES
    )

    # 6. Predict & Save
    print("[5/5] Generating Predictions...")
    preds = model.predict(X_eval_qt)

    # Format dataframe exactly as requested: single column "target01"
    submission_df = pd.DataFrame({'target01': preds})
    
    # Save to CSV
    submission_df.to_csv(OUTPUT_NAME, index=False)
    print("\n" + "="*40)
    print(f"[SUCCESS] Submission file generated: {OUTPUT_NAME}")
    print("You can now upload this file to Moodle.")
    print("="*40)

if __name__ == "__main__":
    generate_submission()
