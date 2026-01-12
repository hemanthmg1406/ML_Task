import os
import json
import hashlib
import joblib
import pandas as pd
import xgboost as xgb
import config
import processing

def generate_fingerprint(final_features):
    """Creates Model DNA for provenance."""
    return {
        "xgb_params": config.XGB_PARAMS,
        "features": final_features,
        "data_hash": hashlib.md5(open(config.DATA_PATH,'rb').read()).hexdigest(),
        "target_hash": hashlib.md5(open(config.TARGET_PATH,'rb').read()).hexdigest(),
        "timestamp": pd.Timestamp.now().isoformat()
    }

def run_production_lock():
    print("!!! INITIATING PRODUCTION MODEL LOCK !!!")
    X, y = processing.load_data(config.DATA_PATH, config.TARGET_PATH)
    
    processor = processing.RankGaussProcessor(config.RANDOM_STATE)
    X_full_qt = processor.fit_transform(X)
    X_full_eng = processing.add_stabilized_interactions(X_full_qt)
    
    target_proc = processing.TargetTransformer()
    y_transformed = target_proc.fit_transform(y)
    
    new_feats = ['feat_189_div_feat_44', 'feat_189_x_feat_44', 'feat_44_x_feat_266', 'feat_189_x_feat_266']
    final_features = sorted(list(set(config.TOP_100_FEATURES + new_feats)))

    print(f"Training on full {len(X)} samples with {len(final_features)} features...")
    model = xgb.XGBRegressor(**config.XGB_PARAMS)
    model.fit(X_full_eng[final_features], y_transformed)

    os.makedirs(config.ARTIFACT_DIR, exist_ok=True)
    
    model.save_model(os.path.join(config.ARTIFACT_DIR, 'trained_model.json'))
    processor.save(config.ARTIFACT_DIR)
    target_proc.save(config.ARTIFACT_DIR) 
    
    joblib.dump(final_features, os.path.join(config.ARTIFACT_DIR, 'feature_list.pkl'))
    
    schema = {"columns": list(X.columns), "dtypes": {c: str(X[c].dtype) for c in X.columns}}
    
    with open(os.path.join(config.ARTIFACT_DIR, 'schema.json'), 'w') as f:
        json.dump(schema, f, indent=2)
    with open(os.path.join(config.ARTIFACT_DIR, 'features.json'), 'w') as f:
        json.dump(final_features, f, indent=2)
    with open(os.path.join(config.ARTIFACT_DIR, 'fingerprint.json'), 'w') as f:
        json.dump(generate_fingerprint(final_features), f, indent=2)

    print("\nSUCCESS: Production artifacts locked in /artifacts.")

if __name__ == "__main__":
    run_production_lock()