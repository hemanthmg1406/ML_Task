import pandas as pd
import os

class DataLoader:
    def __init__(self, data_path, target_path):
        self.data_path = data_path
        self.target_path = target_path

    def load(self):
        """Loads features and target from CSV files."""
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Data file not found: {self.data_path}")
        if not os.path.exists(self.target_path):
            raise FileNotFoundError(f"Target file not found: {self.target_path}")

        print(f"[DataLoader] Loading X from {self.data_path}...")
        X = pd.read_csv(self.data_path)
        
        print(f"[DataLoader] Loading y from {self.target_path}...")
        y = pd.read_csv(self.target_path)
        
        # Ensure y is the correct shape (sometimes it reads as a dataframe)
        if isinstance(y, pd.DataFrame):
            y = y.iloc[:, 0]  # Take the first column as the target series
            
        return X, y

    def select_features(self, X, features):
        """Selects only the specified columns from the dataframe."""
        print(f"[DataLoader] Selecting {len(features)} features...")
        
        # Safety check: Ensure all features exist
        missing_cols = [col for col in features if col not in X.columns]
        if missing_cols:
            raise KeyError(f"The following features are missing from the dataset: {missing_cols}")
            
        return X[features]