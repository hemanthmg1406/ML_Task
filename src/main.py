import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import xgboost as xgb
import config
import processing
from audit import ModelAuditor

def main():
    print("--- STARTING RESEARCH & AUDIT PHASE ---")
    X, y = processing.load_data(config.DATA_PATH, config.TARGET_PATH)
    
    # 1. Audit Split (80/20)
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE
    )

    # 2. Preprocessing Features (Strictly Isolated)
    processor = processing.RankGaussProcessor(config.RANDOM_STATE)
    X_train_qt = processing.add_stabilized_interactions(processor.fit_transform(X_train_raw))
    X_test_qt = processing.add_stabilized_interactions(processor.transform(X_test_raw))
    
    # --- NEW: Initialize and Fit Target Transformer ---
    target_proc = processing.TargetTransformer()
    y_train_t = target_proc.fit_transform(y_train)
    # y_test remains in real units for final validation
    
    new_feats = ['feat_189_div_feat_44', 'feat_189_x_feat_44', 'feat_44_x_feat_266', 'feat_189_x_feat_266']
    final_features = sorted(list(set(config.TOP_30_FEATURES + new_feats)))

    # 3. Training Research Model on Transformed Target
    model = xgb.XGBRegressor(**config.XGB_PARAMS)
    model.fit(X_train_qt[final_features], y_train_t)

    # 4. REPORT RESULTS (Inverting back to Real Units)
    # Training performance check
    train_preds_t = model.predict(X_train_qt[final_features])
    train_preds = target_proc.inverse_transform(train_preds_t)
    train_r2 = r2_score(y_train, train_preds)

    # Testing performance check
    test_preds_t = model.predict(X_test_qt[final_features])
    test_preds = target_proc.inverse_transform(test_preds_t)
    test_r2 = r2_score(y_test, test_preds)

    print("\n" + "="*40)
    print(f"  RESEARCH TRAIN R² (Real Units): {train_r2:.5f}")
    print(f"  RESEARCH TEST  R² (Real Units): {test_r2:.5f}") 
    print("="*40)

    # 5. Run Stress Tests (Passing the Target Transformer to the Auditor)
    # The Auditor now knows how to translate Gaussian model outputs into real-world units
    auditor = ModelAuditor(model, target_proc, config.RANDOM_STATE)
    auditor.run_full_audit(X_train_qt[final_features], y_train, X_test_qt[final_features], y_test)
    
    print("\nAudit Complete. (Artifacts NOT saved in Research Mode)")

if __name__ == "__main__":
    main()