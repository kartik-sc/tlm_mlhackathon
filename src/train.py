import numpy as np
from sklearn.model_selection import StratifiedKFold
from utils import weighted_auc
from model import train_lgb
from config import N_FOLDS, TARGET, WEIGHT, SEED


def run_training(train, test, features, params):
    X = train[features]
    y = train[TARGET]
    w = train[WEIGHT]

    X_test = test[features]

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

    oof_preds = np.zeros(len(X))
    test_preds = np.zeros(len(X_test))

    for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y)):
        print(f"\n===== FOLD {fold+1} =====")

        X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]
        w_tr, w_val = w.iloc[tr_idx], w.iloc[val_idx]

        model = train_lgb(X_tr, y_tr, w_tr, X_val, y_val, w_val, params)

        val_preds = model.predict(X_val)
        oof_preds[val_idx] = val_preds

        test_preds += model.predict(X_test) / N_FOLDS

        fold_score = weighted_auc(y_val, val_preds, w_val)
        print(f"Fold {fold+1} AUC: {fold_score}")

    final_score = weighted_auc(y, oof_preds, w)
    print(f"\n🔥 FINAL CV AUC: {final_score}")

    return test_preds, final_score