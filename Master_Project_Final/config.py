import os

# --- PATHS ---
DATA_PATH = os.path.join('data', 'dataset_29.csv')
TARGET_PATH = os.path.join('data', 'target_29.csv')
ARTIFACT_DIR = 'artifacts'

# --- GLOBAL SETTINGS ---
RANDOM_STATE = 42
TEST_SIZE = 0.2

# --- CHAMPION FEATURES (TOP 30) ---
TOP_30_FEATURES = [
    'feat_44', 'feat_144', 'feat_189', 'feat_200', 'feat_165', 
    'feat_266', 'feat_124', 'feat_101', 'feat_112', 'feat_207', 
    'feat_91', 'feat_163', 'feat_119', 'feat_122', 'feat_203', 
    'feat_175', 'feat_66', 'feat_1', 'feat_148', 'feat_74', 
    'feat_6', 'feat_107', 'feat_263', 'feat_76', 'feat_238', 
    'feat_40', 'feat_212', 'feat_217', 'feat_188', 'feat_115'
]

# --- CHAMPION HYPERPARAMETERS (RELAXED) ---
XGB_PARAMS = {
    'n_estimators': 600,
    'learning_rate': 0.05,
    'max_depth': 5,
    'subsample': 0.72,
    'colsample_bytree': 0.80,
    'reg_alpha': 8.0,
    'reg_lambda': 15.0,
    'min_child_weight': 6,
    'n_jobs': -1,
    'verbosity': 0,
    'random_state': RANDOM_STATE
}
