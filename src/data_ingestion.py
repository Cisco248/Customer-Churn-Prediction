import logging
import pandas as pd
from pathlib import Path
from config import *
from utils.logger import setup_logger

logger = setup_logger()


def load_raw_data(path: Path = RAW_DATA_PATH) -> pd.DataFrame:

    logger.info(f"Loading raw data from: {path}")

    if not path.exists():
        logger.error(f"Raw data not found at {path}")
        raise FileNotFoundError(f"Raw data not found at {path}")

    df = pd.read_csv(path)

    logger.info(f"Loaded {len(df):,} rows | {df.shape[1]} columns")

    return df


def validate_data(df: pd.DataFrame) -> pd.DataFrame:

    missing_cols = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Dataset is Missing Columns: {missing_cols}")

    logger.info("All Required Columns Present")

    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    n_nulls = df["TotalCharges"].isna().sum()
    if n_nulls:
        logger.warning(
            f"TotalCharges: {n_nulls} Values coerced to NaN (will be imputed in preprocessing)"
        )

    null_report = df.isnull().sum()
    if null_report.any():
        logger.info(f"Null Value Report: {null_report[null_report > 0]}")
    else:
        logger.info("No missing values Detected!")

    churn_dist = df[TARGET_COLUMN].value_counts(normalize=True).round(3)
    logger.info(f"Target Distribution: {churn_dist}")

    return df


def run_ingestion() -> pd.DataFrame:
    logger.info("Data Ingestion Started")
    df = load_raw_data()
    logger.info("Row Data Loaded")
    df = validate_data(df)
    logger.info("Data Ingestion Complete")
    return df


if __name__ == "__main__":
    run_ingestion()
