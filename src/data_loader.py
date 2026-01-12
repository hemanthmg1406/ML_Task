import pandas as pd
import os

class DataLoader:
    def __init__(self, data_path, target_path):
        self.data_path = data_path
        self.target_path = target_path

    def load(self):
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"The data file was not found at: {self.data_path}")
        if not os.path.exists(self.target_path):
            raise FileNotFoundError(f"The target file was not found at: {self.target_path}")

        print(f"Loading features from {self.data_path}.")
        X = pd.read_csv(self.data_path)
        
        print(f"Loading target labels from {self.target_path}.")
        y = pd.read_csv(self.target_path)
        
        if isinstance(y, pd.DataFrame):
            y = y.iloc[:, 0]  
            
        return X, y

    def select_features(self, X, features):
        print(f"Selecting {len(features)} specific features from the dataset.")
        
        missing_cols = [col for col in features if col not in X.columns]
        if missing_cols:
            raise KeyError(f"The following features are missing from the dataset: {missing_cols}")
            
        return X[features]