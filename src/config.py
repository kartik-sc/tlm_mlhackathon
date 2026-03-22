SEED = 42
N_FOLDS = 5

TARGET = "label"
WEIGHT = "weight"
ID_COL = "id"

FEATURE_PREFIX = "fx_"

LGBM_PARAMS = {
    "objective": "binary",
    "metric": "auc",
    "learning_rate":0.05,
    "num_leaves": 64,
    "feature_fraction"  : 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "verbosity": -1,
}