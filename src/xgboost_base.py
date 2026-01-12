import os
import joblib
import xgboost as xgb
import config
import processing
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

def run_xgb_baseline():
    print(" XGBOOST BASELINE ")

    X, y = processing.load_data(config.DATA_PATH, config.TARGET_PATH)

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=config.RANDOM_STATE
    )

    feature_list = list(X.columns)

    print(f"Training on {X_train.shape[0]} rows with {len(feature_list)} features")

    model = xgb.XGBRegressor(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=config.RANDOM_STATE,
        n_jobs=-1
    )

    model.fit(
        X_train[feature_list],
        y_train,
        eval_set=[(X_val[feature_list], y_val)],
        verbose=False,
        early_stopping_rounds=50
    )

    preds = model.predict(X_val[feature_list])
    rmse = mean_squared_error(y_val, preds, squared=False)

    print(f"\nBaseline XGB RMSE: {rmse:.5f}")

    os.makedirs(config.ARTIFACT_DIR, exist_ok=True)
    model.save_model(os.path.join(config.ARTIFACT_DIR, "baseline_xgb.json"))
    joblib.dump(feature_list, os.path.join(config.ARTIFACT_DIR, "baseline_features.pkl"))

    print("Baseline model saved.")

if __name__ == "__main__":
    run_xgb_baseline()
