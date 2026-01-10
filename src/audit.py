import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.cluster import KMeans

class ModelAuditor:
    def __init__(self, model, random_state=42):
        self.model = model
        self.random_state = random_state

    def run_full_audit(self, X_train, y_train, X_test, y_test, X_test_qt):
        print("\n" + "="*40)
        print("STARTING SYSTEM AUDIT (CRASH TESTS)")
        print("="*40)
        
        base_pred = self.model.predict(X_test_qt)
        base_r2 = r2_score(y_test, base_pred)
        print(f"[Baseline] Test R²: {base_r2:.4f}")

        self._check_noise(X_test_qt, y_test, base_r2)
        self._check_drift(X_test_qt, y_test)
        self._check_leakage(X_train, y_train)
        self._check_segmentation(X_test_qt, y_test, base_pred)
        print("\n[Audit] Diagnostics Complete.")

    def _check_noise(self, X, y, base_r2):
        print("\n[Test 1] Robustness (Noise Injection 1%)")
        np.random.seed(self.random_state)
        noise = np.random.normal(0, 0.05, X.shape)
        X_noisy = X + noise
        pred = self.model.predict(X_noisy)
        score = r2_score(y, pred)
        drop = base_r2 - score
        print(f"   R² with Noise: {score:.4f} (Drop: {drop:.4f})")
        print(f"   Verdict: {'PASS' if drop < 0.05 else 'FAIL (Fragile)'}")

    def _check_drift(self, X, y):
        print("\n[Test 2] Future Drift (Covariate Shift +0.5 std)")
        X_drift = X + 0.5
        pred = self.model.predict(X_drift)
        score = r2_score(y, pred)
        print(f"   Drifted R²: {score:.4f}")
        print(f"   Verdict: {'PASS' if score > 0.5 else 'FAIL (Sensitive to Shift)'}")

    def _check_leakage(self, X_raw, y):
        print("\n[Test 3] Leakage Hunter")
        X_raw = X_raw.reset_index(drop=True)
        y = y.reset_index(drop=True)
        corrs = X_raw.corrwith(y).abs()
        max_corr = corrs.max()
        print(f"   Max Correlation: {max_corr:.4f}")
        print(f"   Verdict: {'PASS' if max_corr < 0.9 else 'WARNING (Possible Leakage)'}")

    def _check_segmentation(self, X, y, preds):
        print("\n[Test 4] Segmentation Blind Spots")
        kmeans = KMeans(n_clusters=5, random_state=self.random_state, n_init=10)
        clusters = kmeans.fit_predict(X)
        
        df = pd.DataFrame({'y': y, 'pred': preds, 'cluster': clusters})
        df['sq_error'] = (df['y'] - df['pred']) ** 2
        
        rmse_list = []
        for c in range(5):
            rmse = np.sqrt(df[df['cluster']==c]['sq_error'].mean())
            rmse_list.append(rmse)
            
        ratio = max(rmse_list) / min(rmse_list)
        print(f"   Max/Min Error Ratio: {ratio:.2f}")
        print(f"   Verdict: {'PASS' if ratio < 2.0 else 'FAIL (Inconsistent)'}")
