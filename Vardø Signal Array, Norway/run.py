import pandas as pd
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from data_loader import load_data, get_features
from train import run_training
from config import LGB_PARAMS, LGB_PARAM_GRID, ID_COL, TARGET


SEEDS = [42, 52, 62]


def main():
    print("Starting training pipeline...\n")

    # Load data
    train_df, test_df = load_data()
    print("Data loaded")

    # Feature engineering / selection
    features = get_features(train_df)
    print(f"Features ready: {len(features)} features")

    best_score = float("-inf")
    best_params = None
    best_test_preds = None

    # Hyperparameter search loop
    for i, param_set in enumerate(LGB_PARAM_GRID):
        print(f"\n===== CONFIG {i+1} =====")

        params = {**LGB_PARAMS, **param_set}

        seed_test_preds = []
        seed_scores = []

        # 🔥 Seed averaging loop
        for seed in SEEDS:
            print(f"\n--- Seed {seed} ---")

            params["seed"] = seed
            params["bagging_seed"] = seed
            params["feature_fraction_seed"] = seed

            oof_preds, test_preds, score = run_training(
                train_df,
                test_df,
                features,
                params
            )

            seed_test_preds.append(test_preds)
            seed_scores.append(score)

        # Average across seeds
        avg_test_preds = sum(seed_test_preds) / len(seed_test_preds)
        avg_score = sum(seed_scores) / len(seed_scores)

        print(f"\nAvg Score: {avg_score}")

        if avg_score > best_score:
            best_score = avg_score
            best_params = params.copy()
            best_test_preds = avg_test_preds

    print("\n🔥 BEST SCORE:", best_score)
    print("🔥 BEST PARAMS:", best_params)

    print("\n🎉 Training complete!")

    # Save submission
    submission = pd.DataFrame({
        ID_COL: test_df[ID_COL],
        TARGET: best_test_preds
    })

    submission.to_csv("submission.csv", index=False)
    print("✅ Submission saved!")


if __name__ == "__main__":
    main()