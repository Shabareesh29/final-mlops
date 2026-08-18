from dataclasses import dataclass


@dataclass
class DataIngestionConfig:
    train_file_path: str
    test_file_path: str
    rul_file_path: str
    raw_dir: str


@dataclass
class DataValidationConfig:
    train_file_path: str
    test_file_path: str
    rul_file_path: str
    expected_columns: int
    expected_operational_settings: int
    expected_sensors: int


@dataclass
class DataTransformationConfig:
    train_file_path: str
    test_file_path: str
    rul_file_path: str
    processed_dir: str
    scaler_path: str
    rul_cap: int

@dataclass
class ModelTrainerConfig:
    train_file_path: str
    model_path: str
    random_state: int
    n_estimators: int
    max_depth: int
    min_samples_split: int
    min_samples_leaf: int


@dataclass
class ModelEvaluationConfig:
    test_file_path: str
    model_path: str
    predictions_path: str
    max_mae: float
    min_r2: float
    
@dataclass
class PredictionConfig:
    model_path: str
    scaler_path: str
    input_file_path: str
    output_file_path: str

@dataclass
class ModelPusherConfig:
    model_path: str
    model_export_path: str

@dataclass
class CandidateModelConfig:
    source_model_path: str
    candidate_model_path: str