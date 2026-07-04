"""
Data Ingestion Stage
--------------------
Reads the raw Telco churn CSV, validates it against the expected
schema, and writes a validated copy for downstream stages to consume.
This is the first stage in the DVC pipeline.
"""
import os
import pandas as pd

from utils import load_config, get_logger


def ingest_data(config_path: str = "configs/config.yaml") -> pd.DataFrame:
    config = load_config(config_path)
    logger = get_logger("data_ingestion", config["paths"]["log_dir"])

    raw_path = config["data"]["raw_path"]
    ingested_path = config["data"]["ingested_path"]
    required_columns = config["data"]["required_columns"]
    target_column = config["data"]["target_column"]

    # --- Load ---
    if not os.path.exists(raw_path):
        logger.error(f"Raw data file not found at: {raw_path}")
        raise FileNotFoundError(f"Raw data file not found at: {raw_path}")

    logger.info(f"Loading raw data from {raw_path}")
    df = pd.read_csv(raw_path)
    logger.info(f"Raw data shape: {df.shape}")

    # --- Validate ---
    missing_cols = set(required_columns) - set(df.columns)
    if missing_cols:
        logger.error(f"Missing required columns: {missing_cols}")
        raise ValueError(f"Missing required columns: {missing_cols}")

    if df.empty:
        logger.error("Ingested dataframe is empty.")
        raise ValueError("Ingested dataframe is empty.")

    if target_column not in df.columns:
        logger.error(f"Target column '{target_column}' not found.")
        raise ValueError(f"Target column '{target_column}' not found.")

    duplicate_count = df.duplicated().sum()
    if duplicate_count > 0:
        logger.warning(f"Found {duplicate_count} duplicate rows in raw data.")

    logger.info(f"Churn distribution:\n{df[target_column].value_counts(normalize=True)}")

    # --- Save validated copy ---
    os.makedirs(os.path.dirname(ingested_path), exist_ok=True)
    df.to_csv(ingested_path, index=False)
    logger.info(f"Validated data saved to {ingested_path}")

    return df


if __name__ == "__main__":
    ingest_data()