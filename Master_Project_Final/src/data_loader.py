import pandas as pd
import sys

class DataLoader:
    def __init__(self, data_path, target_path):
        self.data_path = data_path
        self.target_path = target_path

    def load(self):
        """Loads data and target CSVs."""
        try:
            print(f"[DataLoader] Loading data from {self.data_path}...")
            X = pd.read_csv(self.data_path)
            y = pd.read_csv(self.target_path)['target01']
            return X, y
        except FileNotFoundError:
            print(f"[Error] Files not found: {self.data_path} or {self.target_path}")
            sys.exit(1)

    @staticmethod
    def select_features(X, feature_list):
        """Returns dataframe with only the selected top features."""
        print(f"[DataLoader] Selecting Top {len(feature_list)} features...")
        return X[feature_list]
