import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
import os
import processing
import config
from sklearn.model_selection import train_test_split

def generate_report_plots():
    print("Generating final diagnostic plots for report...")
    # Load Data and Artifacts
    X, y = processing.load_data(config.DATA_PATH, config.TARGET_PATH)
    artifact_path = config.ARTIFACT_DIR
    
    feat_proc = processing.RankGaussProcessor()
    feat_proc.load(artifact_path)
    
    target_proc = processing.TargetTransformer()
    target_proc.load(artifact_path)
    
    model = xgb.XGBRegressor()
    model.load_model(os.path.join(artifact_path, 'trained_model.json'))
    final_features = joblib.load(os.path.join(artifact_path, 'feature_list.pkl'))
    
    # 80/20 Split for Validation Plotting
    _, X_test_raw, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_test_qt = processing.add_stabilized_interactions(feat_proc.transform(X_test_raw))
    
    # Predict and Inverse Transform
    preds_scaled = model.predict(X_test_qt[final_features])
    preds = target_proc.inverse_transform(preds_scaled)
    residuals = y_test - preds

    # Plot 1: Prediction vs Actual
    plt.figure(figsize=(8, 6))
    sns.regplot(x=y_test, y=preds, scatter_kws={'alpha':0.3}, line_kws={'color':'red'})
    plt.xlabel('Actual Target (Real Units)')
    plt.ylabel('Predicted Target (Real Units)')
    plt.title('Prediction Accuracy and Linearity')
    plt.savefig('plots/final_prediction_accuracy.png')
    plt.close()

    # Plot 2: Residual Plot (Homoscedasticity)
    plt.figure(figsize=(8, 6))
    plt.scatter(preds, residuals, alpha=0.3)
    plt.axhline(0, color='red', linestyle='--')
    plt.xlabel('Predicted Value')
    plt.ylabel('Residual (Error)')
    plt.title('Residual Plot: Homoscedasticity Check')
    plt.savefig('plots/final_residual_plot.png')
    plt.close()

    # Plot 3: Error Distribution
    plt.figure(figsize=(8, 6))
    sns.histplot(residuals, kde=True, bins=30)
    plt.title('Error Distribution (Target Inversion Proof)')
    plt.savefig('plots/final_error_distribution.png')
    plt.close()

    print("Success: Diagnostic plots saved to /plots.")

if __name__ == "__main__":
    os.makedirs('plots', exist_ok=True)
    generate_report_plots()