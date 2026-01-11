import os
import json
import joblib
import pandas as pd
import xgboost as xgb
import processing
import config

def run_evaluation():
    print("Initiating clean-room evaluation...")

    # 1. Load Artifacts (The 'Source of Truth')
    artifact_path = config.ARTIFACT_DIR
    
    with open(os.path.join(artifact_path, 'schema.json'), 'r') as f:
        schema = json.load(f)
        
    # Load the exact feature list used during the production lock
    final_features = joblib.load(os.path.join(artifact_path, 'feature_list.pkl'))
        
    # Load Feature Processor (RankGauss)
    processor = processing.RankGaussProcessor()
    processor.load(artifact_path)
    
    # NEW: Load Target Transformer (Yeo-Johnson)
    target_proc = processing.TargetTransformer()
    target_proc.load(artifact_path)
    
    # Load the XGBoost Model
    model = xgb.XGBRegressor()
    model.load_model(os.path.join(artifact_path, 'trained_model.json'))

    # 2. Load Evaluation Data
    eval_path = os.path.join('data', 'EVAL_29.csv')
    X_eval_raw = pd.read_csv(eval_path)

    # 3. Apply Schema Lock (Ensure column order/presence matches training)
    print("Applying Schema Lock...")
    X_eval_raw = X_eval_raw[schema["columns"]]

    # 4. Transform Features (Using LOADED RankGauss stats)
    print("Applying RankGauss transformation...")
    X_eval_qt = processor.transform(X_eval_raw)

    # 5. Interaction Engineering (Stabilized logic)
    X_eval_eng = processing.add_stabilized_interactions(X_eval_qt)

    # 6. Prediction (Model outputs Gaussian-space values)
    print("Generating predictions from locked model...")
    preds_scaled = model.predict(X_eval_eng[final_features])

    # 7. NEW: The "Un-Stitch" (Inverse Transform back to Real Units)
    # This converts the model's math back into the actual target scale
    print("Inverting target transformation for real-world units...")
    final_preds = target_proc.inverse_transform(preds_scaled)

    # 8. Formatting the final answer
    output = pd.DataFrame({'target': final_preds})

    # 9. Saving with the exact naming convention
    final_output_name = 'EVAL_target01_29.csv'
    output.to_csv(final_output_name, index=False)
    
    print(f"\nSUCCESS: Results inverted and saved to '{final_output_name}'")
    print("The evaluation file is now ready for submission.")

if __name__ == "__main__":
    run_evaluation()