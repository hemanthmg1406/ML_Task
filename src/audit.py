import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.cluster import KMeans

class ModelAuditor:
    def __init__(self, model, target_proc, random_state=42):
        self.model = model
        self.target_proc = target_proc
        self.random_state = random_state

    def _predict_real(self, X):
        preds_scaled = self.model.predict(X)
        return self.target_proc.inverse_transform(preds_scaled)

    def run_full_audit(self, X_train_final, y_train, X_test_final, y_test):
        print("Beginning  audit.")
        
        base_pred = self._predict_real(X_test_final)
        base_r2 = r2_score(y_test, base_pred)
        print(f"The baseline test R² score is {base_r2:.4f}.")

        self._check_noise(X_test_final, y_test, base_r2)
        self._check_drift(X_test_final, y_test)
        
        self._check_segmentation(X_train_final, X_test_final, y_test, base_pred)
        print("The audit complete.")

    def _check_noise(self, X, y, base_r2):
        print("Testing robustness by injecting relative noise.")
        np.random.seed(self.random_state)
        std_vec = X.std(axis=0).values
        noise = np.random.normal(0, 0.05, X.shape) * std_vec
        
        preds_noise = self._predict_real(X + noise)
        score = r2_score(y, preds_noise)
        
        drop = base_r2 - score
        print(f"The R² score after adding noise is {score:.4f}, which is a drop of {drop:.4f}.")
        print(f" result  pass." if drop < 0.05 else "result is fail.")

    def _check_drift(self, X, y):
        print("Check future drift using a relative covariate shift.")
        X_drift = X + (0.5 * X.std(axis=0).values)
        
        preds_drift = self._predict_real(X_drift)
        score = r2_score(y, preds_drift)
        
        print(f"The R² score after the drift is {score:.4f}.")
        print(f"result is a pass." if score > 0.4 else " result is a fail.")

    def _check_segmentation(self, X_train, X_test, y_test, preds):
        kmeans = KMeans(n_clusters=5, random_state=self.random_state, n_init=10)
        kmeans.fit(X_train)
        clusters = kmeans.predict(X_test)
        
        df = pd.DataFrame({'y': y_test, 'pred': preds, 'cluster': clusters})
        df['sq_error'] = (df['y'] - df['pred']) ** 2
        
        rmse_list = [np.sqrt(df[df['cluster']==c]['sq_error'].mean()) for c in range(5)]
        ratio = max(rmse_list) / (min(rmse_list) + 1e-6)
        print(f"The ratio between the maximum and minimum error is {ratio:.2f}.")
        print(f" result is a pass." if ratio < 2.0 else "result is a failu.")