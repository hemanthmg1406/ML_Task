import pandas as pd
import numpy as np
import os
import joblib
from sklearn.preprocessing import QuantileTransformer, PowerTransformer

def load_data(data_path, target_path):
    X = pd.read_csv(data_path)
    y = pd.read_csv(target_path)
    if isinstance(y, pd.DataFrame):
        y = y.iloc[:, 0]
    return X, y

class RankGaussProcessor:
    def __init__(self, random_state=42):
        self.qt = QuantileTransformer(
            n_quantiles=1000, 
            output_distribution='normal', 
            random_state=random_state,
            copy=True
        )

    def fit_transform(self, X):
        return pd.DataFrame(self.qt.fit_transform(X), columns=X.columns, index=X.index)

    def transform(self, X):
        return pd.DataFrame(self.qt.transform(X), columns=X.columns, index=X.index)

    def save(self, directory):
        os.makedirs(directory, exist_ok=True)
        joblib.dump(self.qt, os.path.join(directory, 'quantile_transformer.pkl'))

    def load(self, directory):
        self.qt = joblib.load(os.path.join(directory, 'quantile_transformer.pkl'))

class TargetTransformer:
    def __init__(self):
        self.pt = PowerTransformer(method='yeo-johnson', standardize=True)
        
    def fit_transform(self, y):
        y_reshaped = np.array(y).reshape(-1, 1)
        return self.pt.fit_transform(y_reshaped).ravel()
    
    def transform(self, y):
        y_reshaped = np.array(y).reshape(-1, 1)
        return self.pt.transform(y_reshaped).ravel()
    
    def inverse_transform(self, y_scaled):
        y_reshaped = np.array(y_scaled).reshape(-1, 1)
        return self.pt.inverse_transform(y_reshaped).ravel()
    
    def save(self, directory):
        os.makedirs(directory, exist_ok=True)
        joblib.dump(self.pt, os.path.join(directory, 'target_transformer.pkl'))

    def load(self, directory):
        path = os.path.join(directory, 'target_transformer.pkl')
        if os.path.exists(path):
            self.pt = joblib.load(path)
        else:
            raise FileNotFoundError(f"Target transformer not found at {path}")

def add_stabilized_interactions(df):
    df = df.copy()
    if 'feat_189' in df.columns and 'feat_44' in df.columns:
        df['feat_189_div_feat_44'] = np.tanh(df['feat_189'] / (df['feat_44'] + 1e-6))
        df['feat_189_x_feat_44'] = np.tanh(df['feat_189'] * df['feat_44'])
    if 'feat_44' in df.columns and 'feat_266' in df.columns:
        df['feat_44_x_feat_266'] = np.tanh(df['feat_44'] * df['feat_266'])
    if 'feat_189' in df.columns and 'feat_266' in df.columns:
        df['feat_189_x_feat_266'] = np.tanh(df['feat_189'] * df['feat_266'])
    return df