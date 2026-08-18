import joblib
import pandas as pd

from src.configuration.configuration import ConfigurationManager


class PredictionPipeline:

    def __init__(self, config):

        self.config = config

    # =====================================================
    # PREDICT
    # =====================================================

    def predict(self):

        print("\n" + "=" * 50)
        print("TURBINE RUL PREDICTION")
        print("=" * 50)

        # -------------------------------------------------
        # Load model
        # -------------------------------------------------

        print("\nLoading trained model...")

        model = joblib.load(
            self.config.model_path
        )

        print("✓ Model loaded")

        # -------------------------------------------------
        # Load scaler
        # -------------------------------------------------

        print("Loading scaler...")

        scaler = joblib.load(
            self.config.scaler_path
        )

        print("✓ Scaler loaded")

        # -------------------------------------------------
        # Load input data
        # -------------------------------------------------

        print(
            f"\nLoading input data:"
            f"\n{self.config.input_file_path}"
        )

        df = pd.read_csv(
            self.config.input_file_path
        )

        print(
            f"Input shape: {df.shape}"
        )

        # -------------------------------------------------
        # Rename operational settings
        # -------------------------------------------------

        df = df.rename(
            columns={
                "setting_1": "op_setting_1",
                "setting_2": "op_setting_2",
                "setting_3": "op_setting_3",
            }
        )

        # -------------------------------------------------
        # Feature columns
        # -------------------------------------------------

        feature_columns = [
            "op_setting_1",
            "op_setting_2",
            "op_setting_3",
        ] + [
            f"sensor_{i}"
            for i in range(1, 22)
        ]

        # -------------------------------------------------
        # Check required columns
        # -------------------------------------------------

        required_columns = [
            "unit_number",
            "cycle",
        ] + feature_columns

        missing_columns = [
            column
            for column in required_columns
            if column not in df.columns
        ]

        if missing_columns:

            raise ValueError(
                "Missing required columns: "
                + ", ".join(missing_columns)
            )

        # -------------------------------------------------
        # Scale features
        # -------------------------------------------------

        print("\nScaling input features...")

        scaled_features = scaler.transform(
            df[feature_columns]
        )

        scaled_features = pd.DataFrame(
            scaled_features,
            columns=feature_columns,
            index=df.index,
        )

        print("✓ Features scaled")

        # -------------------------------------------------
        # Create model input
        # -------------------------------------------------

        model_input = pd.concat(
            [
                df[["cycle"]],
                scaled_features,
            ],
            axis=1,
        )

        print(
            f"Model input shape:"
            f" {model_input.shape}"
        )

        # -------------------------------------------------
        # Generate predictions
        # -------------------------------------------------

        print("\nGenerating predictions...")

        predictions = model.predict(
            model_input
        )

        # RUL cannot be negative
        predictions = predictions.clip(
            min=0
        )

        print("✓ Predictions generated")

        # -------------------------------------------------
        # Create results
        # -------------------------------------------------

        results = df[
            ["unit_number", "cycle"]
        ].copy()

        results["predicted_rul"] = predictions

        # -------------------------------------------------
        # Save predictions
        # -------------------------------------------------

        output_path = self.config.output_file_path

        output_dir = output_path.rsplit(
            "\\",
            1
        )[0] if "\\" in output_path else output_path.rsplit(
            "/",
            1
        )[0]

        import os

        os.makedirs(
            output_dir,
            exist_ok=True
        )

        results.to_csv(
            output_path,
            index=False
        )

        # -------------------------------------------------
        # Display results
        # -------------------------------------------------

        print("\n" + "=" * 50)
        print("PREDICTION RESULTS")
        print("=" * 50)

        print("\nFirst 10 predictions:\n")

        print(
            results.head(10).to_string(
                index=False
            )
        )

        print(
            f"\nPredictions saved to:"
            f"\n{output_path}"
        )

        print("\n" + "=" * 50)
        print("PREDICTION COMPLETED")
        print("=" * 50)

        return results


# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":

    configuration_manager = (
        ConfigurationManager()
    )

    prediction_config = (
        configuration_manager
        .get_prediction_config()
    )

    prediction_pipeline = (
        PredictionPipeline(
            config=prediction_config
        )
    )

    prediction_pipeline.predict()