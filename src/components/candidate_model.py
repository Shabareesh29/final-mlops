from pathlib import Path
import shutil

from src.entity.config_entity import CandidateModelConfig


class CandidateModel:

    def __init__(self, config: CandidateModelConfig):

        self.config = config

    def create_candidate(self):

        print("\n" + "=" * 50)
        print("CANDIDATE MODEL")
        print("=" * 50)

        source_model = Path(
            self.config.source_model_path
        )

        candidate_model = Path(
            self.config.candidate_model_path
        )

        if not source_model.exists():

            raise FileNotFoundError(
                f"Trained model not found: "
                f"{source_model}"
            )

        candidate_model.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        shutil.copy2(
            source_model,
            candidate_model
        )

        print(
            f"\nCandidate model created:"
            f"\n{candidate_model}"
        )

        print("\n✓ Candidate model ready")

        print("\n" + "=" * 50)
        print("CANDIDATE MODEL CREATED")
        print("=" * 50)

        return candidate_model