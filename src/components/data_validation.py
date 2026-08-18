from pathlib import Path

import pandas as pd

from src.entity.config_entity import DataValidationConfig


class DataValidation:

    def __init__(self, config: DataValidationConfig):
        self.config = config

    def validate_files(self):

        print("\n" + "=" * 50)
        print("DATA VALIDATION")
        print("=" * 50)

        files = {
            "Training": self.config.train_file_path,
            "Testing": self.config.test_file_path,
            "RUL": self.config.rul_file_path,
        }

        # ---------------------------------------------
        # Check files
        # ---------------------------------------------

        for name, file_path in files.items():

            if not Path(file_path).exists():

                print(f"❌ {name} file not found:")
                print(file_path)

                return False

            print(f"✓ {name} file found")

        # ---------------------------------------------
        # Load data
        # ---------------------------------------------

        train_df = pd.read_csv(
            self.config.train_file_path,
            sep=r"\s+",
            header=None,
        )

        test_df = pd.read_csv(
            self.config.test_file_path,
            sep=r"\s+",
            header=None,
        )

        rul_df = pd.read_csv(
            self.config.rul_file_path,
            header=None,
        )

        # ---------------------------------------------
        # Column validation
        # ---------------------------------------------

        expected_columns = self.config.expected_columns

        if train_df.shape[1] != expected_columns:

            print(
                f"❌ Training dataset has "
                f"{train_df.shape[1]} columns."
            )

            return False

        if test_df.shape[1] != expected_columns:

            print(
                f"❌ Test dataset has "
                f"{test_df.shape[1]} columns."
            )

            return False

        print(f"✓ Training columns: {train_df.shape[1]}")
        print(f"✓ Test columns: {test_df.shape[1]}")

        # ---------------------------------------------
        # Missing values
        # ---------------------------------------------

        train_missing = train_df.isna().sum().sum()
        test_missing = test_df.isna().sum().sum()
        rul_missing = rul_df.isna().sum().sum()

        print(f"✓ Training missing values: {train_missing}")
        print(f"✓ Test missing values: {test_missing}")
        print(f"✓ RUL missing values: {rul_missing}")

        if train_missing > 0:
            print("❌ Training data contains missing values.")
            return False

        if test_missing > 0:
            print("❌ Test data contains missing values.")
            return False

        if rul_missing > 0:
            print("❌ RUL data contains missing values.")
            return False

        # ---------------------------------------------
        # Engine validation
        # ---------------------------------------------

        train_engines = train_df[0].nunique()
        test_engines = test_df[0].nunique()
        rul_count = len(rul_df)

        print(f"✓ Training engines: {train_engines}")
        print(f"✓ Test engines: {test_engines}")
        print(f"✓ RUL values: {rul_count}")

        if test_engines != rul_count:

            print(
                "❌ Test engine count does not "
                "match RUL count."
            )

            return False

        # ---------------------------------------------
        # Numeric validation
        # ---------------------------------------------

        if not train_df.apply(
            lambda column: pd.api.types.is_numeric_dtype(column)
        ).all():

            print("❌ Training data contains non-numeric columns.")

            return False

        if not test_df.apply(
            lambda column: pd.api.types.is_numeric_dtype(column)
        ).all():

            print("❌ Test data contains non-numeric columns.")

            return False

        print("✓ Training data types valid")
        print("✓ Test data types valid")

        # ---------------------------------------------
        # Final result
        # ---------------------------------------------

        print("\n" + "=" * 50)
        print("DATA VALIDATION PASSED")
        print("=" * 50)

        return True