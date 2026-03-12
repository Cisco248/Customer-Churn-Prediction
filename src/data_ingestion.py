import pandas as pd
from pathlib import Path
from config import REQUIRED_COLS, TARGET_COLUMN, RAW_DATA_PATH
from utils.logger import setup_logger


class DataIngestion:
    def __init__(self, path: Path):
        self.logger = setup_logger()
        self.location = path

    def load_dataset(self) -> pd.DataFrame:
        self.logger.info("🚀 ===> Data_Ingestion Stage: Start Processing")

        if not self.location.exists():
            self.logger.error(f"Raw data not found at {self.location} ===> ❌")
            raise ValueError(f"Raw data not found at {self.location} ===> ❌")

        self.df = pd.read_csv(self.location)

        self.logger.info(
            f"DataFrame: {len(self.df):,} rows | {self.df.shape[1]} columns ===> ℹ️"
        )
        self.logger.info("✅ ===> Data_Ingestion Stage: Completed Processing")

        return self.df

    def validate_data(self) -> pd.DataFrame:

        self.logger.info("🚀 ===> Data_Ingestion Stage: Start Validating")

        self.missing_cols = [c for c in REQUIRED_COLS if c not in self.df.columns]

        if self.missing_cols:
            self.logger.error(
                f"Dataset is Missing Columns: {self.missing_cols} ===> ❌"
            )
            raise ValueError(f"Dataset is Missing Columns: {self.missing_cols} ===> ❌")

        self.df["TotalCharges"] = pd.to_numeric(
            self.df["TotalCharges"], errors="coerce"
        )

        self.n_nulls = self.df["TotalCharges"].isna().sum()
        if self.n_nulls:
            self.logger.warning(
                f"⚠️ ===> Null Coerrence in TotalCharges: {self.n_nulls}"
            )

        self.null_report = self.df.isnull().sum()
        if self.null_report.any():
            self.logger.warning(
                f"⚠️ ===> Null Value Report: {self.null_report[self.null_report > 0].to_string()}"
            )
        else:
            self.logger.info("No missing values Detected! ===> ✅")

        self.churn_dist = self.df[TARGET_COLUMN].value_counts(normalize=True).round(3)
        self.logger.info(
            f"Target Distribution: {self.churn_dist['Yes']:.3%} Churn | {self.churn_dist['No']:.3%} No Churn ===> ℹ️"
        )

        self.logger.info("✅ ===> Data_Ingestion Stage: Completed Validating")

        return self.df
