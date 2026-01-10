# src/processing.py
import pandas as pd
import joblib
import os
from sklearn.preprocessing import QuantileTransformer

class RankGaussProcessor:
    def __init__(self, random_state=42):
        self.scaler = QuantileTransformer(output_distribution='normal', random_state=random_state)
        self.columns = None

    def fit(self, X):
        self.columns = X.columns
        self.scaler.fit(X)

    def transform(self, X):
        if self.columns is None:
            raise ValueError("Processor must be fit before transform.")
        
        X_trans = self.scaler.transform(X)
        return pd.DataFrame(X_trans, columns=self.columns)

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

    def save(self, directory):
        """Saves the fitted scaler."""
        os.makedirs(directory, exist_ok=True)
        path = os.path.join(directory, 'preprocessor.pkl')
        joblib.dump(self.scaler, path)
        print(f"[Processor] Saved to {path}")