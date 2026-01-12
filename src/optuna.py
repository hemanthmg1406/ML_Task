import optuna
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import config
import processing
import numpy as np

def objective(trial):
    X, y = processing.load_data(config.DATA_PATH, config.TARGET_PATH)
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE
    )

    feat_proc = processing.RankGaussProcessor(random_state=config.RANDOM_STATE)
    target_proc = processing.TargetTransformer()
    
    X_train_qt = processing.add_stabilized_interactions(feat_proc.fit_transform(X_train_raw))
    X_test_qt = processing.add_stabilized_interactions(feat_proc.transform(X_test_raw))
    
    y_train_t = target_proc.fit_transform(y_train)
    
    new_feats = ['feat_189_div_feat_44', 'feat_189_x_feat_44', 'feat_44_x_feat_266', 'feat_189_x_feat_266']
    feats = sorted(list(set(config.TOP_100_FEATURES + new_feats)))

    params = {
        'n_estimators': trial.suggest_int('n_estimators', 1000, 3000),
        'max_depth': trial.suggest_int('max_depth', 3, 5),          
        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.03, log=True),
        'min_child_weight': trial.suggest_int('min_child_weight', 50, 200), 
        'subsample': trial.suggest_float('subsample', 0.5, 0.7),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.3, 0.5), 
        'reg_alpha': trial.suggest_float('reg_alpha', 0.5, 20.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 20.0, 150.0, log=True), 
        'random_state': config.RANDOM_STATE,
        'n_jobs': -1
    }

    model = xgb.XGBRegressor(**params)
    model.fit(X_train_qt[feats], y_train_t)
    
    X_val = X_test_qt[feats]
    X_std = X_val.std(axis=0).values
    
    def predict_real(data):
        return target_proc.inverse_transform(model.predict(data))

    base_preds = predict_real(X_val)
    base_r2 = r2_score(y_test, base_preds)
    
    noise_data = X_val + 0.01 * np.random.normal(0, 1, X_val.shape) * X_std
    noise_r2 = r2_score(y_test, predict_real(noise_data))
    
    drift_data = X_val + 0.5 * X_std
    drift_r2 = r2_score(y_test, predict_real(drift_data))

    noise_drop = max(0, base_r2 - noise_r2)
    drift_drop = max(0, base_r2 - drift_r2)
    
    score = base_r2 - (0.5 * noise_drop) - (1.5 * drift_drop)
    
    return max(-1.0, score)

if __name__ == "__main__":
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=100, n_jobs=1) 

    print("\nBest Stable-100 Parameters Found:")
    print(study.best_trial.params)