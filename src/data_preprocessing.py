"""
Data Preprocessing Stage
-------------------------
Takes the validated raw data and applies cleaning + encoding:
- Binary encode Yes/No and Male/Female columns
- One-hot encode multi-category columns
- Clean TotalCharges (coerce to numeric, fill NaN with 0)
- Drop customerID
- Convert bool columns to int
This is the second stage in the DVC pipeline.
"""
import os
import pandas as pd

pd.set_option('future.no_silent_downcasting', True)

from utils import load_config, get_logger


BINARY_COLS = ["gender", "Partner", "Dependents", "PhoneService", "PaperlessBilling", "Churn"]

MULTI_CAT_COLS = [
    "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
    "Contract", "PaymentMethod"
]


def preprocess_data(config_path: str = "configs/config.yaml") -> pd.DataFrame:
    config = load_config(config_path)
    logger = get_logger("data_preprocessing", config["paths"]["log_dir"])

    ingested_path = config["data"]["ingested_path"]
    preprocessed_path = "data/interim/preprocessed_telco_churn.csv"

    if not os.path.exists(ingested_path):
        logger.error(f"Ingested data not found at: {ingested_path}")
        raise FileNotFoundError(f"Ingested data not found at: {ingested_path}")

    logger.info(f"Loading ingested data from {ingested_path}")
    df = pd.read_csv(ingested_path)
    logger.info(f"Input shape: {df.shape}")

    # --- Binary encoding ---
    df[BINARY_COLS] = df[BINARY_COLS].replace({
    "Yes": 1, "No": 0,
    "Male": 1, "Female": 0
}).infer_objects(copy=False)
    logger.info(f"Binary encoded columns: {BINARY_COLS}")

    # --- One-hot encoding ---
    df = pd.get_dummies(df, columns=MULTI_CAT_COLS, drop_first=True)
    logger.info(f"One-hot encoded columns: {MULTI_CAT_COLS}")

    # --- Clean TotalCharges ---
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    nan_count = df["TotalCharges"].isna().sum()
    if nan_count > 0:
        logger.warning(f"Found {nan_count} NaNs in TotalCharges (tenure=0 customers). Filling with 0.")
        df["TotalCharges"] = df["TotalCharges"].fillna(0)

    # --- Drop customerID ---
    df = df.drop("customerID", axis=1)

    # --- Convert bool columns to int (from one-hot encoding) ---
    bool_cols = df.select_dtypes(include="bool").columns
    df[bool_cols] = df[bool_cols].astype(int)

    logger.info(f"Output shape: {df.shape}")

    # --- Save ---
    os.makedirs(os.path.dirname(preprocessed_path), exist_ok=True)
    df.to_csv(preprocessed_path, index=False)
    logger.info(f"Preprocessed data saved to {preprocessed_path}")

    return df


if __name__ == "__main__":
    preprocess_data()