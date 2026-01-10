# main.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# Import Modules
import config
from src.data_loader import DataLoader
from src.processing import RankGaussProcessor
from src.model import ChampionModel
from src.audit import ModelAuditor

def main():
    print("   MASTER'S PROJECT: FINAL PIPELINE     ")

    # 1. Load Data
    loader = DataLoader(config.DATA_PATH, config.TARGET_PATH)
    X_raw, y_raw = loader.load()
    X_selected = loader.select_features(X_raw, config.TOP_30_FEATURES)

    # 2. Split for Audit (80/20)
    print("\n[Step 2] Splitting data for Validation Audit...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_selected, y_raw, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE
    )

    # 3. Preprocessing (Fit on Train, Transform Test)
    print("\n[Step 3] Applying RankGauss Transformation...")
    processor = RankGaussProcessor(config.RANDOM_STATE)
    X_train_qt = processor.fit_transform(X_train)
    X_test_qt = processor.transform(X_test)

    # 4. Train Champion Model
    print("\n[Step 4] Training Champion XGBoost Model...")
    model = ChampionModel(config.XGB_PARAMS)
    model.train(X_train_qt, y_train)

    # 5. Run Safety Audit
    auditor = ModelAuditor(model, config.RANDOM_STATE)
    auditor.run_full_audit(X_train, y_train, X_test, y_test, X_test_qt)

    # 6. Final Production Training (Full Data)
    print("\n" + "="*40)
    print("FINAL STEP: PRODUCTION DEPLOYMENT")
    print("="*40)
    print("Retraining on 100% of data...")
    
    # Fit processor on ALL data
    final_processor = RankGaussProcessor(config.RANDOM_STATE)
    X_full_qt = final_processor.fit_transform(X_selected)
    
    # Train model on ALL data
    final_model = ChampionModel(config.XGB_PARAMS)
    final_model.train(X_full_qt, y_raw)
    
    # Save Artifacts
    print("\nSaving Artifacts...")
    final_model.save(config.ARTIFACT_DIR)
    final_processor.save(config.ARTIFACT_DIR)
    
    print("\n[SUCCESS] Pipeline Complete. Model ready for submission.")

if __name__ == "__main__":
    main()  