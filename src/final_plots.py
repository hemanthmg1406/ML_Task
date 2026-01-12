import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
import os
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance
from sklearn.metrics import r2_score, mean_squared_error

# Import your local modules
import config
import processing

# Ensure output directory exists
OUTPUT_DIR = "report_plots_v2"
os.makedirs(OUTPUT_DIR, exist_ok=True)
# Set style for professional reports
plt.style.use('seaborn-v0_8-whitegrid')

def get_data_ready():
    """Helper to load and preprocess data cleanly."""
    print("Loading data...")
    X, y = processing.load_data(config.DATA_PATH, config.TARGET_PATH)
    
    if hasattr(config, 'TOP_100_FEATURES'):
        features = config.TOP_100_FEATURES
    else:
        features = config.TOP_30_FEATURES
        
    X = X[features]
    return X, y, features

# ==========================================
# 5. RankGauss Before/After (Visualizing Preprocessing)
# ==========================================
def plot_rankgauss_demonstration(X):
    print("\n[5] Generating RankGauss Before/After Plot...")
    
    target_feat = 'feat_44'
    if target_feat not in X.columns:
        target_feat = X.columns[0]
        
    raw_data = X[target_feat].values
    
    rg = processing.RankGaussProcessor(config.RANDOM_STATE)
    df_single = pd.DataFrame({target_feat: raw_data})
    transformed_data = rg.fit_transform(df_single)[target_feat].values
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=False)
    
    # Left: Raw
    sns.histplot(raw_data, bins=50, kde=True, ax=axes[0], color='#E24A33', edgecolor='black', alpha=0.6)
    axes[0].set_title(f'BEFORE: Raw Feature Distribution\n(Skewed, Non-Normal)', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Raw Feature Value')
    axes[0].set_ylabel('Frequency')
    
    # Right: RankGauss
    sns.histplot(transformed_data, bins=50, kde=True, ax=axes[1], color='#348ABD', edgecolor='black', alpha=0.6)
    axes[1].set_title('AFTER: RankGauss Transformation\n(Forced Normal Distribution)', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Standard Deviations (Gaussian Scale)')
    axes[1].set_ylabel('Frequency')
    
    # Annotation
    plt.figtext(0.5, 0.01, f"Feature: {target_feat} | N={len(raw_data)} samples | Method: QuantileTransformer(output_distribution='normal')", 
                ha="center", fontsize=10, bbox={"facecolor":"white", "alpha":0.5, "pad":5})
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15) # Make room for text
    save_path = os.path.join(OUTPUT_DIR, '05_rankgauss_transform_v2.png')
    plt.savefig(save_path, dpi=300)
    print(f"   Saved to {save_path}")

# ==========================================
# 2. Learning Curve (Check for Overfitting)
# ==========================================
def plot_learning_curve(X, y):
    print("\n[2] Generating Learning Curve...")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE
    )
    
    proc = processing.RankGaussProcessor(config.RANDOM_STATE)
    X_train_qt = proc.fit_transform(X_train)
    X_test_qt = proc.transform(X_test)
    
    target_proc = processing.TargetTransformer()
    y_train_t = target_proc.fit_transform(y_train)
    y_test_t = target_proc.transform(y_test)
    
    model = xgb.XGBRegressor(**config.XGB_PARAMS)
    
    eval_set = [(X_train_qt, y_train_t), (X_test_qt, y_test_t)]
    model.fit(X_train_qt, y_train_t, eval_set=eval_set, verbose=False)
    
    results = model.evals_result()
    epochs = len(results['validation_0']['rmse'])
    x_axis = range(0, epochs)
    
    train_rmse = results['validation_0']['rmse']
    val_rmse = results['validation_1']['rmse']
    
    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(x_axis, train_rmse, label=f'Training RMSE (Final: {train_rmse[-1]:.4f})', color='gray', linestyle='--')
    plt.plot(x_axis, val_rmse, label=f'Validation RMSE (Final: {val_rmse[-1]:.4f})', color='#E24A33', linewidth=2)
    
    # Annotation: Best Iteration
    best_iter = np.argmin(val_rmse)
    min_rmse = val_rmse[best_iter]
    
    plt.axvline(best_iter, color='black', linestyle=':', alpha=0.5)
    plt.scatter(best_iter, min_rmse, color='black', s=50, zorder=5)
    plt.annotate(f'Best Fit\nIter: {best_iter}\nRMSE: {min_rmse:.4f}', 
                 xy=(best_iter, min_rmse), xytext=(best_iter+50, min_rmse+0.05),
                 arrowprops=dict(facecolor='black', shrink=0.05))

    plt.legend()
    plt.ylabel('RMSE (Transformed Units)')
    plt.xlabel('Boosting Rounds')
    plt.title('Learning Curve: Generalization Gap Analysis', fontsize=14)
    
    save_path = os.path.join(OUTPUT_DIR, '02_learning_curve_v2.png')
    plt.savefig(save_path, dpi=300)
    print(f"   Saved to {save_path}")
    return model, proc, target_proc, X_test, y_test 

# ==========================================
# 1. Permutation Importance (Stability Check)
# ==========================================
def plot_permutation_importance(model, X_test_raw, y_test, feat_proc, target_proc):
    print("\n[1] Generating Permutation Importance...")
    
    X_test_qt = feat_proc.transform(X_test_raw)
    y_test_t = target_proc.transform(y_test)
    
    result = permutation_importance(
        model, X_test_qt, y_test_t,
        n_repeats=5, random_state=config.RANDOM_STATE, n_jobs=-1, scoring='neg_root_mean_squared_error'
    )
    
    sorted_idx = result.importances_mean.argsort()
    top_n = 20
    top_idx = sorted_idx[-top_n:]
    
    plt.figure(figsize=(12, 8))
    
    # Create Boxplot
    box = plt.boxplot(
        result.importances[top_idx].T,
        vert=False,
        labels=X_test_qt.columns[top_idx],
        patch_artist=True
    )
    
    # Style boxes
    for patch in box['boxes']:
        patch.set_facecolor('#348ABD')
        patch.set_alpha(0.6)
        
    # Vertical Reference Line at 0
    plt.axvline(0, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label='Zero Importance (Noise)')
    
    # Annotate Top 3 Means
    for i in range(1, 4):
        idx = top_idx[-i] 
        mean_val = result.importances_mean[idx]
        y_pos = top_n - i + 1
        plt.text(mean_val, y_pos, f" Mean: {mean_val:.4f}", va='center', fontsize=9, fontweight='bold', color='#333')

    plt.title(f'Top {top_n} Features: Permutation Importance (Test Set)', fontsize=14)
    plt.xlabel('RMSE Increase when Feature is Scrambled')
    plt.legend(loc='lower right')
    plt.tight_layout()
    
    save_path = os.path.join(OUTPUT_DIR, '01_permutation_importance_v2.png')
    plt.savefig(save_path, dpi=300)
    print(f"   Saved to {save_path}")

# ==========================================
# 3. Predicted vs Actual (Real Units Check)
# ==========================================
def plot_pred_vs_actual(model, X_test_raw, y_test, feat_proc, target_proc):
    print("\n[3] Generating Predicted vs Actual...")
    
    X_test_qt = feat_proc.transform(X_test_raw)
    preds_scaled = model.predict(X_test_qt)
    preds_real = target_proc.inverse_transform(preds_scaled)
    
    # Calc Stats
    r2 = r2_score(y_test, preds_real)
    rmse = np.sqrt(mean_squared_error(y_test, preds_real))
    
    plt.figure(figsize=(8, 8))
    
    # Scatter with alpha
    plt.scatter(y_test, preds_real, alpha=0.3, color='#348ABD', edgecolor='k', s=20, label='Test Predictions')
    
    # Identity Line (Perfect Prediction)
    min_val = min(y_test.min(), preds_real.min())
    max_val = max(y_test.max(), preds_real.max())
    plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', linewidth=2, label='Perfect Fit (y=x)')
    
    # Stats Box
    stats_text = f"$R^2$: {r2:.4f}\nRMSE: {rmse:.4f}"
    plt.gca().text(0.05, 0.95, stats_text, transform=plt.gca().transAxes,
                   fontsize=12, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.xlabel('Actual Target (Real Units)')
    plt.ylabel('Predicted Target (Real Units)')
    plt.title('Model Accuracy: Predicted vs Actual', fontsize=14)
    plt.legend(loc='lower right')
    plt.grid(True, linestyle=':', alpha=0.6)
    
    save_path = os.path.join(OUTPUT_DIR, 'final_prediction_accuracy_v2.png')
    plt.savefig(save_path, dpi=300)
    print(f"   Saved to {save_path}")
    
    return r2 

# ==========================================
# 4. Robustness/Noise Plot (Stability Stress Test)
# ==========================================
def plot_noise_robustness(model, X_test_raw, y_test, feat_proc, target_proc, base_r2):
    print("\n[4] Generating Noise Robustness Plot...")
    
    noise_levels = np.linspace(0, 0.2, 10)
    r2_scores = []
    
    X_test_qt = feat_proc.transform(X_test_raw)
    std_vec = X_test_qt.std(axis=0).values
    
    for noise in noise_levels:
        if noise == 0:
            r2_scores.append(base_r2)
            continue
            
        np.random.seed(config.RANDOM_STATE)
        noise_matrix = np.random.normal(0, noise, X_test_qt.shape) * std_vec
        X_test_noisy = X_test_qt + noise_matrix
        
        preds_scaled = model.predict(X_test_noisy)
        preds_real = target_proc.inverse_transform(preds_scaled)
        
        score = r2_score(y_test, preds_real)
        r2_scores.append(score)

    # Plot
    plt.figure(figsize=(9, 6))
    plt.plot(noise_levels, r2_scores, marker='o', color='purple', linewidth=2, label='R² Degradation')
    
    # Threshold Line (90% of base)
    threshold = base_r2 * 0.9
    plt.axhline(threshold, color='red', linestyle='--', label='90% Reliability Threshold')
    
    # Annotate Baseline
    plt.annotate(f'Baseline R²: {base_r2:.4f}', xy=(0, base_r2), xytext=(0.02, base_r2 + 0.02),
                 arrowprops=dict(facecolor='black', arrowstyle='->'))
    
    # Annotate Drop at specific noise
    noise_point = 0.05
    idx = (np.abs(noise_levels - noise_point)).argmin()
    score_at_noise = r2_scores[idx]
    drop = base_r2 - score_at_noise
    
    plt.annotate(f'Drop @ 5% Noise: -{drop:.4f}', 
                 xy=(noise_levels[idx], score_at_noise), 
                 xytext=(noise_levels[idx], score_at_noise - 0.05),
                 arrowprops=dict(facecolor='red', shrink=0.05))

    plt.xlabel('Noise Level (Standard Deviations)')
    plt.ylabel('R² Score (Real Units)')
    plt.title('System Robustness: Sensitivity to Input Noise', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    save_path = os.path.join(OUTPUT_DIR, '02_noise_robustness_v2.png')
    plt.savefig(save_path, dpi=300)
    print(f"   Saved to {save_path}")

if __name__ == "__main__":
    X, y, features = get_data_ready()
    
    # Plot 5: RankGauss
    plot_rankgauss_demonstration(X)
    
    # Plot 2: Learning Curve
    model, proc, target_proc, X_test, y_test = plot_learning_curve(X, y)
    
    # Plot 1: Permutation
    plot_permutation_importance(model, X_test, y_test, proc, target_proc)
    
    # Plot 3: Predicted vs Actual (Returns R2 for the next plot)
    base_r2 = plot_pred_vs_actual(model, X_test, y_test, proc, target_proc)
    
    # Plot 4: Robustness
    plot_noise_robustness(model, X_test, y_test, proc, target_proc, base_r2)
    
    print(f"\nSUCCESS: All publication-ready plots saved to local folder: /{OUTPUT_DIR}")