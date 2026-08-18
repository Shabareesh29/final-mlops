from pathlib import Path

import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.entity.config_entity import DataTransformationConfig


class DataTransformation:

    def __init__(self, config: DataTransformationConfig):
        self.config = config

        self.processed_dir = Path(config.processed_dir)
        self.processed_dir.mkdir(
            parents=True,
            exist_ok=True
        )

    def _get_column_names(self):

        return [
            "unit_number",
            "cycle",
            "op_setting_1",
            "op_setting_2",
            "op_setting_3",
            "sensor_1",
            "sensor_2",
            "sensor_3",
            "sensor_4",
            "sensor_5",
            "sensor_6",
            "sensor_7",
            "sensor_8",
            "sensor_9",
            "sensor_10",
            "sensor_11",
            "sensor_12",
            "sensor_13",
            "sensor_14",
            "sensor_15",
            "sensor_16",
            "sensor_17",
            "sensor_18",
            "sensor_19",
            "sensor_20",
            "sensor_21",
        ]

    def _load_data(self):

        columns = self._get_column_names()

        train_df = pd.read_csv(
            self.config.train_file_path,
            sep=r"\s+",
            header=None,
            names=columns,
        )

        test_df = pd.read_csv(
            self.config.test_file_path,
            sep=r"\s+",
            header=None,
            names=columns,
        )

        rul_df = pd.read_csv(
            self.config.rul_file_path,
            header=None,
            names=["rul"],
        )

        return train_df, test_df, rul_df

    def _calculate_train_rul(self, train_df):

        max_cycle = train_df.groupby(
            "unit_number"
        )["cycle"].transform("max")

        train_df["rul"] = max_cycle - train_df["cycle"]

        train_df["rul"] = train_df["rul"].clip(
            upper=self.config.rul_cap
        )

        return train_df

    def _calculate_test_rul(self, test_df, rul_df):

    # RUL_FD004 contains one RUL value for each test engine.
    # Engine IDs are 1...248, while pandas uses 0...247
    # as the default index.

        rul_mapping = pd.Series(
        rul_df["rul"].values,
        index=range(1, len(rul_df) + 1)
    )

        test_max_cycle = test_df.groupby(
        "unit_number"
    )["cycle"].transform("max")

        actual_rul = test_df["unit_number"].map(
        rul_mapping
    )

        test_df["rul"] = (
        test_max_cycle
        - test_df["cycle"]
        + actual_rul
    )

        test_df["rul"] = test_df["rul"].clip(
        upper=self.config.rul_cap
    )

        return test_df

    def _remove_constant_features(
        self,
        train_df,
        test_df
    ):

        feature_columns = [
            column
            for column in train_df.columns
            if column not in [
                "unit_number",
                "cycle",
                "rul",
            ]
        ]

        constant_features = [
            column
            for column in feature_columns
            if train_df[column].nunique() <= 1
        ]

        if constant_features:

            print(
                "\nRemoving constant features:"
            )

            for feature in constant_features:
                print(f"  - {feature}")

            train_df = train_df.drop(
                columns=constant_features
            )

            test_df = test_df.drop(
                columns=constant_features
            )

        return train_df, test_df

    def transform(self):

        print("\n" + "=" * 50)
        print("DATA TRANSFORMATION")
        print("=" * 50)

        # -----------------------------------------
        # Load data
        # -----------------------------------------

        train_df, test_df, rul_df = self._load_data()

        print(
            f"Original train shape: {train_df.shape}"
        )

        print(
            f"Original test shape: {test_df.shape}"
        )

        # -----------------------------------------
        # Calculate RUL
        # -----------------------------------------

        train_df = self._calculate_train_rul(
            train_df
        )

        test_df = self._calculate_test_rul(
            test_df,
            rul_df
        )

        print("\n✓ Training RUL calculated")
        print("✓ Test RUL calculated")

        # -----------------------------------------
        # Remove constant features
        # -----------------------------------------

        train_df, test_df = (
            self._remove_constant_features(
                train_df,
                test_df
            )
        )

        # -----------------------------------------
        # Feature columns
        # -----------------------------------------

        feature_columns = [
            column
            for column in train_df.columns
            if column not in [
                "unit_number",
                "cycle",
                "rul",
            ]
        ]

        print(
            f"\nFeatures before scaling: "
            f"{len(feature_columns)}"
        )

        # -----------------------------------------
        # Scaling
        # -----------------------------------------

        scaler = StandardScaler()

        train_df[feature_columns] = scaler.fit_transform(
            train_df[feature_columns]
        )

        test_df[feature_columns] = scaler.transform(
            test_df[feature_columns]
        )

        # -----------------------------------------
        # Save scaler
        # -----------------------------------------

        scaler_path = Path(
            self.config.scaler_path
        )

        scaler_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        joblib.dump(
            scaler,
            scaler_path
        )

        # -----------------------------------------
        # Save processed datasets
        # -----------------------------------------

        train_output = (
            self.processed_dir
            / "train_processed.csv"
        )

        test_output = (
            self.processed_dir
            / "test_processed.csv"
        )

        train_df.to_csv(
            train_output,
            index=False
        )

        test_df.to_csv(
            test_output,
            index=False
        )

        # -----------------------------------------
        # Summary
        # -----------------------------------------

        print(
            f"\nProcessed train shape: "
            f"{train_df.shape}"
        )

        print(
            f"Processed test shape: "
            f"{test_df.shape}"
        )

        print(
            f"Scaler saved to: "
            f"{scaler_path}"
        )

        print(
            f"Training data saved to: "
            f"{train_output}"
        )

        print(
            f"Test data saved to: "
            f"{test_output}"
        )

        print("\n" + "=" * 50)
        print("DATA TRANSFORMATION COMPLETED")
        print("=" * 50)

        return train_df, test_df