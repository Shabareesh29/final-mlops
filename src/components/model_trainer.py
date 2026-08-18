from pathlib import Path
import mlflow
import joblib
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

from src.entity.config_entity import ModelTrainerConfig


class ModelTrainer:

    def __init__(self, config: ModelTrainerConfig):
        self.config = config

    def train(self):

        print("\n" + "=" * 50)
        print("MODEL TRAINING")
        print("=" * 50)

        # -----------------------------------------
        # Load processed training data
        # -----------------------------------------

        train_df = pd.read_csv(
            self.config.train_file_path
        )

        print(
            f"Training data shape: {train_df.shape}"
        )

        # -----------------------------------------
        # Separate features and target
        # -----------------------------------------

        target_column = "rul"

        X = train_df.drop(
            columns=[
                "rul",
                "unit_number",
            ]
        )

        y = train_df[target_column]

        print(
            f"Features used for training: {X.shape[1]}"
        )

        print(
            f"Training samples: {X.shape[0]}"
        )

        # -----------------------------------------
        # Train/validation split
        # -----------------------------------------

        X_train, X_validation, y_train, y_validation = (
            train_test_split(
                X,
                y,
                test_size=0.2,
                random_state=self.config.random_state,
            )
        )

        print(
            f"Training split: {X_train.shape}"
        )

        print(
            f"Validation split: {X_validation.shape}"
        )

        # -----------------------------------------
        # Create Random Forest
        # -----------------------------------------

        model = RandomForestRegressor(
            n_estimators=self.config.n_estimators,
            max_depth=self.config.max_depth,
            min_samples_split=self.config.min_samples_split,
            min_samples_leaf=self.config.min_samples_leaf,
            random_state=self.config.random_state,
            n_jobs=-1,
        )

        # -----------------------------------------
        # Train
        # -----------------------------------------

        print("\nTraining Random Forest...")

        model.fit(
            X_train,
            y_train
        )

        print("✓ Model training completed")

        # -----------------------------------------
        # Validation prediction
        # -----------------------------------------

        predictions = model.predict(
            X_validation
        )

        validation_mae = mean_absolute_error(
            y_validation,
            predictions
        )
        mlflow.log_metric(
            "validation_mae",
            validation_mae
        )

        print(
            f"\nValidation MAE: "
            f"{validation_mae:.4f}"
        )

        # -----------------------------------------
        # Save model
        # -----------------------------------------

        model_path = Path(
            self.config.model_path
        )

        model_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        joblib.dump(
            model,
            model_path
        )
        mlflow.log_artifact(
            str(model_path),
            artifact_path="model"
        )

        print(
            f"\nModel saved to: "
            f"{model_path}"
        )

        print("\n" + "=" * 50)
        print("MODEL TRAINING COMPLETED")
        print("=" * 50)

        return model