import os

DATA_PATH = os.path.join('data', 'dataset_29.csv')
TARGET_PATH = os.path.join('data', 'target_29.csv')
ARTIFACT_DIR = 'artifacts'

RANDOM_STATE = 42
TEST_SIZE = 0.2

TOP_100_FEATURES = [
    "feat_44", "feat_144", "feat_189", "feat_266", "feat_124", "feat_200", "feat_165", "feat_101", 
    "feat_112", "feat_207", "feat_122", "feat_163", "feat_206", "feat_107", "feat_62", "feat_184", 
    "feat_100", "feat_92", "feat_91", "feat_203", "feat_243", "feat_202", "feat_236", "feat_263", 
    "feat_134", "feat_13", "feat_249", "feat_74", "feat_212", "feat_253", "feat_125", "feat_76", 
    "feat_88", "feat_141", "feat_6", "feat_81", "feat_213", "feat_196", "feat_75", "feat_262", 
    "feat_148", "feat_135", "feat_85", "feat_127", "feat_232", "feat_58", "feat_48", "feat_24", 
    "feat_59", "feat_108", "feat_29", "feat_68", "feat_69", "feat_155", "feat_56", "feat_57", 
    "feat_65", "feat_73", "feat_123", "feat_10", "feat_208", "feat_105", "feat_230", "feat_156", 
    "feat_204", "feat_93", "feat_46", "feat_119", "feat_67", "feat_121", "feat_118", "feat_251", 
    "feat_110", "feat_219", "feat_245", "feat_37", "feat_20", "feat_139", "feat_221", "feat_183", 
    "feat_168", "feat_39", "feat_120", "feat_170", "feat_152", "feat_173", "feat_31", "feat_260", 
    "feat_181", "feat_30", "feat_218", "feat_42", "feat_70", "feat_226", "feat_17", "feat_22", 
    "feat_63", "feat_19", "feat_227", "feat_131"
]

XGB_PARAMS = {
    'n_estimators': 2029,
    'max_depth': 6,
    'learning_rate': 0.005036228781320912,
    'min_child_weight': 52,
    'subsample': 0.6135651933764772,
    'colsample_bytree': 0.48325177403940434,
    'reg_alpha': 0.10054602637816404,
    'reg_lambda': 15.5607790765264,
    'n_jobs': -1,
    'random_state': RANDOM_STATE
}