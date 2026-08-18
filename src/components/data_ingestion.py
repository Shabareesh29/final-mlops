import pandas as pd
from pathlib import Path

from src.entity.config_entity import DataIngestionConfig


class DataIngestion:

    def __init__(self,
                 config: DataIngestionConfig):

        self.config = config

    def load_raw_data(self):

        train_df = pd.read_csv(
            self.config.train_file_path,
            sep=r"\s+",
            header=None
        )

        test_df = pd.read_csv(
            self.config.test_file_path,
            sep=r"\s+",
            header=None
        )

        rul_df = pd.read_csv(
            self.config.rul_file_path,
            header=None
        )

        print("\nTRAIN SHAPE:", train_df.shape)
        print("TEST SHAPE:", test_df.shape)
        print("RUL SHAPE:", rul_df.shape)

        return train_df, test_df, rul_df