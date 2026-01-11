import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.cluster import KMeans

class ModelAuditor:
    def __init__(self, model, target_proc, random_state=42):
        self.model = model
        self.target_proc = target_proc  # Auditor now translates Gaussian -> Real Units
        self.random_state = random_state

    def _predict_real(self, X):
        """Helper to predict in scaled space and immediately invert to real units."""
        preds_scaled = self.model.predict(X)
        return self.target_proc.inverse_transform(preds_scaled)

    def run_full_audit(self, X_train_final, y_train, X_test_final, y_test):
        print("\n" + "="*40)
        print("STARTING PHYSICALLY CORRECT SYSTEM AUDIT")
        print("="*40)
        
        # Real-world units prediction
        base_pred = self._predict_real(X_test_final)
        base_r2 = r2_score(y_test, base_pred)
        print(f"[Baseline] Test R²: {base_r2:.4f}")

        # Testing robustness in real units
        self._check_noise(X_test_final, y_test, base_r2)
        self._check_drift(X_test_final, y_test)
        
        # Segmentation check
        self._check_segmentation(X_train_final, X_test_final, y_test, base_pred)
        print("\n[Audit] Diagnostics Complete.")

    def _check_noise(self, X, y, base_r2):
        print("\n[Test 1] Robustness (Relative Noise Injection)")
        np.random.seed(self.random_state)
        std_vec = X.std(axis=0).values
        noise = np.random.normal(0, 0.05, X.shape) * std_vec
        
        # Use inverse-transformed predictions
        preds_noise = self._predict_real(X + noise)
        score = r2_score(y, preds_noise)
        
        drop = base_r2 - score
        print(f"   R² with Noise: {score:.4f} (Drop: {drop:.4f})")
        print(f"   Verdict: {'PASS' if drop < 0.05 else 'FAIL (Fragile)'}")

    def _check_drift(self, X, y):
        print("\n[Test 2] Future Drift (Relative Covariate Shift +0.5 std)")
        X_drift = X + (0.5 * X.std(axis=0).values)
        
        # Use inverse-transformed predictions
        preds_drift = self._predict_real(X_drift)
        score = r2_score(y, preds_drift)
        
        print(f"   Drifted R²: {score:.4f}")
        print(f"   Verdict: {'PASS' if score > 0.4 else 'FAIL'}") # Adjusted threshold for honest R2

    def _check_segmentation(self, X_train, X_test, y_test, preds):
        print("\n[Test 4] Segmentation Blind Spots (Leakage-Free)")
        kmeans = KMeans(n_clusters=5, random_state=self.random_state, n_init=10)
        kmeans.fit(X_train)
        clusters = kmeans.predict(X_test)
        
        df = pd.DataFrame({'y': y_test, 'pred': preds, 'cluster': clusters})
        df['sq_error'] = (df['y'] - df['pred']) ** 2
        
        rmse_list = [np.sqrt(df[df['cluster']==c]['sq_error'].mean()) for c in range(5)]
        ratio = max(rmse_list) / (min(rmse_list) + 1e-6)
        print(f"   Max/Min Error Ratio: {ratio:.2f}")
        print(f"   Verdict: {'PASS' if ratio < 2.0 else 'FAIL'}")