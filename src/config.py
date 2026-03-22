SEED = 42
N_FOLDS = 5

TARGET = "label"
WEIGHT = "weight"
ID_COL = "id"

FEATURE_PREFIX = "fx_"

LGB_PARAMS = {
    "objective": "binary",
    "metric": "auc",
    "learning_rate": 0.05,
    "num_leaves": 256,
    "feature_fraction": 0.8,
    "min_data_in_leaf": 20,
    "lambda_l2": 0,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "verbosity": -1,
}

LGB_PARAM_GRID = [
    {
        "learning_rate": 0.05,
        "num_leaves": 64,
        "min_data_in_leaf": 20,
        "lambda_l2": 0,
    },
    {
        "learning_rate": 0.05,
        "num_leaves": 128,
        "min_data_in_leaf": 50,
        "lambda_l2": 1,
    },
    {
        "learning_rate": 0.03,
        "num_leaves": 128,
        "min_data_in_leaf": 100,
        "lambda_l2": 5,
    },
]