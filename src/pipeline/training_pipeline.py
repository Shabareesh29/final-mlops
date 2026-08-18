import dagshub
import mlflow

from src.configuration.configuration import ConfigurationManager
from src.components.data_ingestion import DataIngestion
from src.components.data_validation import DataValidation
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.components.model_evaluation import ModelEvaluation
from src.components.model_pusher import ModelPusher
from src.components.candidate_model import CandidateModel


def run_pipeline():

    print("\n" + "=" * 60)
    print("STARTING MLOPS PIPELINE")
    print("=" * 60)

    # =====================================================
    # CONFIGURATION
    # =====================================================

    config = ConfigurationManager()

    # =====================================================
    # DAGSHUB + MLFLOW
    # =====================================================

    dagshub.init(
        repo_owner="kshabareesh78",
        repo_name="final-mlops",
        mlflow=True,
    )

    experiment_name = config.config["mlflow"]["experiment_name"]

    mlflow.set_experiment(experiment_name)

    # =====================================================
    # START ONE MLFLOW RUN
    # =====================================================

    with mlflow.start_run(
        run_name="fd004-random-forest-pipeline"
    ) as run:

        print("\nMLflow Run ID:")
        print(run.info.run_id)

        # =================================================
        # DATA INGESTION
        # =================================================

        print("\n" + "=" * 50)
        print("DATA INGESTION")
        print("=" * 50)

        ingestion_config = (
            config.get_data_ingestion_config()
        )

        ingestion = DataIngestion(
            ingestion_config
        )

        ingestion.load_raw_data()

        print("✓ Data ingestion completed")

        # =================================================
        # DATA VALIDATION
        # =================================================

        validation_config = (
            config.get_data_validation_config()
        )

        validation = DataValidation(
            validation_config
        )

        validation_passed = (
            validation.validate_files()
        )

        if not validation_passed:

            print("\n❌ Data validation failed.")
            print("❌ Pipeline stopped.")

            mlflow.set_tag(
                "pipeline_status",
                "validation_failed"
            )

            return

        print("✓ Data validation completed")

        # =================================================
        # DATA TRANSFORMATION
        # =================================================

        transformation_config = (
            config.get_data_transformation_config()
        )

        transformation = DataTransformation(
            transformation_config
        )

        transformation.transform()

        print("✓ Data transformation completed")

        # =================================================
        # MODEL TRAINING CONFIGURATION
        # =================================================

        model_config = (
            config.get_model_trainer_config()
        )

        # Log model parameters to MLflow

        mlflow.log_params(
            {
                "model_type": "RandomForestRegressor",
                "random_state": model_config.random_state,
                "n_estimators": model_config.n_estimators,
                "max_depth": model_config.max_depth,
                "min_samples_split": (
                    model_config.min_samples_split
                ),
                "min_samples_leaf": (
                    model_config.min_samples_leaf
                ),
                "dataset": "NASA CMAPSS FD004",
            }
        )

        # =================================================
        # MODEL TRAINING
        # =================================================

        model_trainer = ModelTrainer(
            model_config
        )

        model = model_trainer.train()

        print("✓ Model training completed")

        # =================================================
        # CANDIDATE MODEL
        # =================================================

        print("\n" + "=" * 50)
        print("CREATING CANDIDATE MODEL")
        print("=" * 50)

        candidate_config = (
            config.get_candidate_model_config()
        )

        candidate_model = CandidateModel(
            candidate_config
        )

        candidate_model.create_candidate()

        print("\n✓ Candidate model created")

                # =================================================
        # MODEL EVALUATION
        # =================================================

        evaluation_config = (
            config.get_model_evaluation_config()
        )

        evaluator = ModelEvaluation(
            evaluation_config
        )

        evaluation_results = evaluator.evaluate()

        print("✓ Model evaluation completed")

        # =================================================
        # MODEL PUSHER
        # =================================================

    if evaluation_results["approved"]:

        print("\n" + "=" * 50)
        print("MODEL APPROVED")
        print("=" * 50)

        model_pusher_config = (
            config.get_model_pusher_config()
        )

        model_pusher = ModelPusher(
            model_pusher_config
        )

        model_pusher.push_model()

    else:

        print("\n" + "=" * 50)
        print("MODEL REJECTED")
        print("=" * 50)

        print(
        "\nCandidate model did not pass "
        "the quality gate."
    )

        print(
        "Production model was NOT changed."
    )
        # =================================================
        # PIPELINE STATUS
        # =================================================

        mlflow.set_tag(
            "pipeline_status",
            "training_completed"
        )

        mlflow.set_tag(
            "dataset",
            "NASA CMAPSS FD004"
        )

        mlflow.set_tag(
            "model",
            "Random Forest Regressor"
        )

        print("\n" + "=" * 60)
        print("PIPELINE COMPLETED SUCCESSFULLY")
        print("=" * 60)

        print("\nMLflow Run ID:")
        print(run.info.run_id)

        print(
            "\nMLflow Run URL:"
        )

        print(
            f"https://dagshub.com/"
            f"kshabareesh78/final-mlops.mlflow/"
            f"#/experiments/0/runs/{run.info.run_id}"
        )


if __name__ == "__main__":
    run_pipeline()