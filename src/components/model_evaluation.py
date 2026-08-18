from pathlib import Path

import joblib
import mlflow
import pandas as pd

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


class ModelEvaluation:

    def __init__(self, config):

        self.config = config

    def evaluate(self):

        print("\n" + "=" * 50)
        print("MODEL EVALUATION")
        print("=" * 50)

        # =================================================
        # LOAD TEST DATA
        # =================================================

        test_df = pd.read_csv(
            self.config.test_file_path
        )

        print(
            f"\nTest data shape: "
            f"{test_df.shape}"
        )

        # =================================================
        # LOAD CANDIDATE MODEL
        # =================================================

        print("\nLoading candidate model...")

        model = joblib.load(
            self.config.model_path
        )

        print("✓ Candidate model loaded")

        # =================================================
        # SELECT FINAL CYCLE OF EACH ENGINE
        # =================================================

        final_cycle_df = (
            test_df
            .sort_values(
                ["unit_number", "cycle"]
            )
            .groupby(
                "unit_number"
            )
            .tail(1)
            .copy()
        )

        print(
            f"\nTest engines evaluated: "
            f"{len(final_cycle_df)}"
        )

        # =================================================
        # FEATURES AND TARGET
        # =================================================

        X_test = final_cycle_df.drop(
            columns=[
                "rul",
                "unit_number",
            ]
        )

        y_test = final_cycle_df[
            "rul"
        ]

        # =================================================
        # GENERATE PREDICTIONS
        # =================================================

        print("\nGenerating predictions...")

        predictions = model.predict(
            X_test
        )

        print("✓ Predictions generated")

        # =================================================
        # CALCULATE METRICS
        # =================================================

        mae = mean_absolute_error(
            y_test,
            predictions
        )

        rmse = mean_squared_error(
            y_test,
            predictions
        ) ** 0.5

        r2 = r2_score(
            y_test,
            predictions
        )

        # =================================================
        # LOG METRICS TO MLFLOW
        # =================================================

        mlflow.log_metrics(
            {
                "test_mae": mae,
                "test_rmse": rmse,
                "test_r2": r2,
            }
        )

        # =================================================
        # MODEL QUALITY GATE
        # =================================================

        print("\n" + "=" * 50)
        print("MODEL QUALITY GATE")
        print("=" * 50)

        print(
            f"\nMaximum allowed MAE: "
            f"{self.config.max_mae}"
        )

        print(
            f"Minimum required R²: "
            f"{self.config.min_r2}"
        )

        print(
            f"\nActual MAE: "
            f"{mae:.4f}"
        )

        print(
            f"Actual R²: "
            f"{r2:.4f}"
        )

        # Model must satisfy BOTH conditions
        model_approved = (
            mae <= self.config.max_mae
            and
            r2 >= self.config.min_r2
        )

        # =================================================
        # APPROVED
        # =================================================

        if model_approved:

            print(
                "\n✓ MODEL QUALITY GATE PASSED"
            )

            print(
                "✓ Candidate model is approved"
            )

            mlflow.set_tag(
                "model_status",
                "approved"
            )

        # =================================================
        # REJECTED
        # =================================================

        else:

            print(
                "\n❌ MODEL QUALITY GATE FAILED"
            )

            print(
                "❌ Candidate model is rejected"
            )

            mlflow.set_tag(
                "model_status",
                "rejected"
            )

        # =================================================
        # CREATE PREDICTION DATAFRAME
        # =================================================

        predictions_df = pd.DataFrame(
            {
                "unit_number":
                    final_cycle_df[
                        "unit_number"
                    ].values,

                "actual_rul":
                    y_test.values,

                "predicted_rul":
                    predictions,
            }
        )

        predictions_df["error"] = (
            predictions_df["actual_rul"]
            -
            predictions_df["predicted_rul"]
        )

        # =================================================
        # SAVE PREDICTIONS
        # =================================================

        predictions_path = Path(
            self.config.predictions_path
        )

        predictions_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        predictions_df.to_csv(
            predictions_path,
            index=False
        )

        print(
            f"\nPredictions saved to:"
            f"\n{predictions_path}"
        )

        # =================================================
        # LOG PREDICTIONS TO MLFLOW
        # =================================================

        mlflow.log_artifact(
            str(predictions_path),
            artifact_path="evaluation"
        )

        # =================================================
        # PRINT FINAL RESULTS
        # =================================================

        print("\n" + "=" * 50)
        print("EVALUATION RESULTS")
        print("=" * 50)

        print(
            f"\nMAE:  {mae:.4f}"
        )

        print(
            f"RMSE: {rmse:.4f}"
        )

        print(
            f"R²:   {r2:.4f}"
        )

        print(
            f"\nModel approved: "
            f"{model_approved}"
        )

        print("\n" + "=" * 50)
        print("MODEL EVALUATION COMPLETED")
        print("=" * 50)

        return {
            "mae": mae,
            "rmse": rmse,
            "r2": r2,
            "approved": model_approved,
        }