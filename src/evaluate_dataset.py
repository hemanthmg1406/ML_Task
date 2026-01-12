import os
import json
import joblib
import pandas as pd
import xgboost as xgb
import processing
import config

def run_evaluation():
    print("Starting the model evaluation process.")

    artifact_path = config.ARTIFACT_DIR
    
    with open(os.path.join(artifact_path, 'schema.json'), 'r') as f:
        schema = json.load(f)
        
    final_features = joblib.load(os.path.join(artifact_path, 'feature_list.pkl'))
        
    processor = processing.RankGaussProcessor()
    processor.load(artifact_path)
    
    target_proc = processing.TargetTransformer()
    target_proc.load(artifact_path)
    
    model = xgb.XGBRegressor()
    model.load_model(os.path.join(artifact_path, 'trained_model.json'))

    eval_path = os.path.join('data', 'EVAL_29.csv')
    X_eval_raw = pd.read_csv(eval_path)

    print("Verifying the dataset schema.")
    X_eval_raw = X_eval_raw[schema["columns"]]

    print("Transforming features using the RankGauss method.")
    X_eval_qt = processor.transform(X_eval_raw)

    X_eval_eng = processing.add_stabilized_interactions(X_eval_qt)

    print("Calculating predictions from the trained model.")
    preds_scaled = model.predict(X_eval_eng[final_features])

    print("Converting predictions back to real-world units.")
    final_preds = target_proc.inverse_transform(preds_scaled)

    output = pd.DataFrame({'target': final_preds})

    final_output_name = 'EVAL_target01_29.csv'
    output.to_csv(final_output_name, index=False)
    
    print(f"The results have been successfully saved to {final_output_name}.")
    print("The evaluation results are ready for submission.")

if __name__ == "__main__":
    run_evaluation()