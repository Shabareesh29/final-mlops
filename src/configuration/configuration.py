import yaml

from src.entity.config_entity import (
    DataIngestionConfig,
    DataValidationConfig,
    DataTransformationConfig,
    ModelTrainerConfig,
    ModelEvaluationConfig,
)


class ConfigurationManager:

    def __init__(self, config_filepath="config/model.yaml"):

        with open(config_filepath, "r") as file:
            self.config = yaml.safe_load(file)

    # =================================================
    # DATA INGESTION
    # =================================================

    def get_data_ingestion_config(self):

        return DataIngestionConfig(
            train_file_path=self.config["dataset"]["train_file"],
            test_file_path=self.config["dataset"]["test_file"],
            rul_file_path=self.config["dataset"]["rul_file"],
            raw_dir=self.config["artifacts"]["raw_dir"],
        )

    # =================================================
    # DATA VALIDATION
    # =================================================

    def get_data_validation_config(self):

        return DataValidationConfig(
            train_file_path=self.config["dataset"]["train_file"],
            test_file_path=self.config["dataset"]["test_file"],
            rul_file_path=self.config["dataset"]["rul_file"],
            expected_columns=self.config["validation"]["expected_columns"],
            expected_operational_settings=self.config[
                "validation"
            ]["expected_operational_settings"],
            expected_sensors=self.config["validation"]["expected_sensors"],
        )

    # =================================================
    # DATA TRANSFORMATION
    # =================================================

    def get_data_transformation_config(self):

        return DataTransformationConfig(
            train_file_path=self.config["dataset"]["train_file"],
            test_file_path=self.config["dataset"]["test_file"],
            rul_file_path=self.config["dataset"]["rul_file"],
            processed_dir=self.config["artifacts"]["processed_dir"],
            scaler_path=self.config["artifacts"]["scaler_path"],
            rul_cap=self.config["transformation"]["rul_cap"],
        )

    # =================================================
    # MODEL TRAINING
    # =================================================

    def get_model_trainer_config(self):

        return ModelTrainerConfig(
            train_file_path=(
                self.config["artifacts"]["processed_dir"]
                + "/train_processed.csv"
            ),

            model_path=self.config["artifacts"]["model_path"],

            random_state=self.config["model"]["random_state"],

            n_estimators=(
                self.config["model"]["random_forest"]["n_estimators"]
            ),

            max_depth=(
                self.config["model"]["random_forest"]["max_depth"]
            ),

            min_samples_split=(
                self.config["model"]["random_forest"]["min_samples_split"]
            ),

            min_samples_leaf=(
                self.config["model"]["random_forest"]["min_samples_leaf"]
            ),
        )

    def get_model_evaluation_config(self):

        return ModelEvaluationConfig(
        test_file_path=(
            self.config["artifacts"]["processed_dir"]
            + "/test_processed.csv"
        ),

        model_path=self.config["artifacts"]["model_path"],

        predictions_path=self.config["evaluation"][
            "predictions_path"
        ],
    )