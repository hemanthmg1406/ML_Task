# Applied Machine Learning and Regression Analysis

This repository contains the code and methodology for **Work Package 1** of my applied machine learning project. The main goal of this work is to build a robust **regression pipeline using XGBoost** that generalizes well to unseen data.

The raw dataset initially contained **highly skewed features and extreme outliers**, which caused a baseline XGBoost model to lose accuracy under even small amounts of noise. To address this, the project emphasizes **careful feature engineering, stability, and rigorous model evaluation**, with a strong focus on preventing data leakage and overfitting while tuning the most impactful machine learning parameters.

---

## Pipeline Overview

### 1. Feature Preprocessing

The raw input features were heavily skewed and contained extreme values. To handle this:

- **RankGauss normalization** was applied to map feature ranks onto a standard normal distribution while preserving their relative ordering.
- This transformation significantly reduces the influence of outliers and heavy-tailed distributions.
- Additional **stabilized interaction features** were created using transformations such as `log1p` and `tanh` to prevent extreme values from dominating model behavior.

---

### 2. Target Variable Stabilization

The target variable was also highly skewed. To improve learning stability:

- A **Yeo-Johnson power transformation** was applied to make the target distribution more symmetric.
- This leads to a smoother and more symmetric error surface, enabling faster and more stable convergence.
- During evaluation, predictions are **inverse-transformed** back to their original scale so results remain interpretable.

---

### 3. Feature Selection

The dataset contained many redundant or unstable signals that encouraged memorization rather than generalization. To mitigate this:

- A **stability-based feature selection** approach was implemented.
- The method runs **bootstrapped permutation importance** across multiple data splits.
- Features are ranked using a custom **stability score**, defined as the ratio of a feature’s mean importance to its variance.
- Only consistently informative features are retained for training.

---

### 4. Model Training and Optimization

- The final model is an **XGBoost regressor** trained exclusively on the stable feature subset.
- All preprocessing steps are fit **only on the training data** and then applied to validation splits to fully avoid data leakage.
- **Optuna** is used for hyperparameter optimization.
- The tuning objective explicitly penalizes configurations whose performance degrades under:
  - 1% feature noise
  - Minor covariate shifts  
- This encourages higher regularization and more robust model behavior.

---

## Repository Structure

- `src/main.py`  
  Runs the complete pipeline: data loading, preprocessing, model training, and final evaluation.

- `src/features.py`  
  Performs stability-based feature selection using permutation importance and stores the selected features.

- `src/xgboost_base.py`  
  Baseline XGBoost model trained on raw features for comparison purposes.

- `src/config.py`  
  Central configuration file containing file paths, random seeds, selected features, and optimized model parameters.

- `requirements.txt`  
  Python dependencies required to run the project.

---

## How to Run

Install the required dependencies:

```bash
pip install -r requirements.txt
