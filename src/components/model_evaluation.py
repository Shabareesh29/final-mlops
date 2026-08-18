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

        # =========================================
        # Load test data
        # =========================================

        test_df = pd.read_csv(
            self.config.test_file_path
        )

        print(
            f"Test data shape: {test_df.shape}"
        )

        # =========================================
        # Load trained model
        # =========================================

        print("\nLoading trained model...")

        model = joblib.load(
            self.config.model_path
        )

        print("✓ Model loaded")

        # =========================================
        # Select final cycle of each engine
        # =========================================

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

        # =========================================
        # Features and target
        # =========================================

        X_test = final_cycle_df.drop(
            columns=[
                "rul",
                "unit_number",
            ]
        )

        y_test = final_cycle_df["rul"]

        # =========================================
        # Prediction
        # =========================================

        print("\nGenerating predictions...")

        predictions = model.predict(
            X_test
        )

        print("✓ Predictions generated")

        # =========================================
        # Metrics
        # =========================================

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

        # =========================================
        # Log metrics to MLflow
        # =========================================

        mlflow.log_metrics(
            {
                "test_mae": mae,
                "test_rmse": rmse,
                "test_r2": r2,
            }
        )

        # =========================================
        # Create prediction dataframe
        # =========================================

        predictions_df = pd.DataFrame(
            {
                "unit_number": final_cycle_df[
                    "unit_number"
                ].values,

                "actual_rul": y_test.values,

                "predicted_rul": predictions,

            }
        )

        predictions_df["error"] = (
            predictions_df["actual_rul"]
            - predictions_df["predicted_rul"]
        )

        # =========================================
        # Save predictions
        # =========================================

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

        # =========================================
        # Log predictions artifact
        # =========================================

        mlflow.log_artifact(
            str(predictions_path),
            artifact_path="evaluation"
        )

        # =========================================
        # Print results
        # =========================================

        print("\n" + "=" * 50)
        print("UNSEEN DATA EVALUATION")
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
            f"\nPredictions saved to:"
            f"\n{predictions_path}"
        )

        print("\n" + "=" * 50)
        print("MODEL EVALUATION COMPLETED")
        print("=" * 50)

        return {
            "mae": mae,
            "rmse": rmse,
            "r2": r2,
        }