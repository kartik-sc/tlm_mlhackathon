import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from data_loader import load_data, get_features
from train import run_training
from config import LGB_PARAMS, LGB_PARAM_GRID, ID_COL


def main():
    print("Starting training pipeline...\n")

    # Load data
    train_df, test_df = load_data()
    print("Data loaded")

    # Feature engineering / selection
    features = get_features(train_df)
    print(f"Features ready: {len(features)} features")

    best_score = 0
    best_params = None
    best_test_preds = None

    # Hyperparameter search loop
    for i, param_set in enumerate(LGB_PARAM_GRID):
        print(f"\n===== CONFIG {i+1} =====")

        params = {**LGB_PARAMS, **param_set}

        oof_preds, test_preds, score = run_training(
            train_df,
            test_df,
            features,
            params
        )

        print(f"Score: {score}")

        if score > best_score:
            best_score = score
            best_params = params
            best_test_preds = test_preds

    print("\n🔥 BEST SCORE:", best_score)
    print("🔥 BEST PARAMS:", best_params)

    print("\n🎉 Training complete!")


if __name__ == "__main__":
    main()