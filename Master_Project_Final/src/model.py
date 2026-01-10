import xgboost as xgb
import joblib
import os

class ChampionModel:
    def __init__(self, params):
        self.model = xgb.XGBRegressor(**params)

    def train(self, X, y):
        print(f"[Model] Training XGBoost with {self.model.n_estimators} trees...")
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def save(self, directory):
        os.makedirs(directory, exist_ok=True)
        path = os.path.join(directory, 'champion_model.json')
        self.model.save_model(path)
        print(f"[Model] Saved to {path}")
