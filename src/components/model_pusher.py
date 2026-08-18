from pathlib import Path
import shutil

from src.entity.config_entity import ModelPusherConfig


class ModelPusher:

    def __init__(self, config: ModelPusherConfig):

        self.config = config

    def push_model(self):

        print("\n" + "=" * 50)
        print("MODEL PUSHER")
        print("=" * 50)

        source_model = Path(
            self.config.model_path
        )

        production_model = Path(
            self.config.model_export_path
        )

        # ---------------------------------------------
        # Check candidate model
        # ---------------------------------------------

        if not source_model.exists():

            raise FileNotFoundError(
                f"Candidate model not found: "
                f"{source_model}"
            )

        print(
            f"\nCandidate model:"
            f"\n{source_model}"
        )

        # ---------------------------------------------
        # Create production directory
        # ---------------------------------------------

        production_model.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        # ---------------------------------------------
        # Push candidate to production
        # ---------------------------------------------

        shutil.copy2(
            source_model,
            production_model
        )

        print(
            f"\nProduction model:"
            f"\n{production_model}"
        )

        print(
            "\n✓ Model pushed to production"
        )

        print("\n" + "=" * 50)
        print("MODEL PUSHER COMPLETED")
        print("=" * 50)

        return production_model