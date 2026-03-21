from data_loader import load_data, get_features
from trainer import run_training
from config import LGB_PARAMS, ID_COL
import pandas as pd
import os

os.makedirs("outputs/submissions", exist_ok=True)

def main():
    train, test = load_data()
    features = get_features(train)

    test_preds, score = run_training(train, test, features, LGB_PARAMS)

    submission = pd.DataFrame({
        ID_COL: test[ID_COL],
        "label": test_preds
    })

    submission.to_csv("outputs/submissions/submission.csv", index=False)
    print("Submission saved!")


if __name__ == "__main__":
    main()