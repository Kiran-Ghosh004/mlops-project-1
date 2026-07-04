"""
Feature Engineering Stage
--------------------------
Takes the preprocessed data and applies feature engineering:
- Collapses redundant 'No internet service' dummy columns (from OnlineSecurity,
  OnlineBackup, DeviceProtection, TechSupport, StreamingTV, StreamingMovies)
  into a single 'No_internet_service' flag.
- Collapses 'MultipleLines_No phone service' into 'No_phone_service'.
This addresses the multicollinearity identified via VIF in the EDA notebook.
This is the third stage in the DVC pipeline.
"""
import os
import pandas as pd

from utils import load_config, get_logger


def engineer_features(config_path: str = "configs/config.yaml") -> pd.DataFrame:
    config = load_config(config_path)
    logger = get_logger("feature_engineering", config["paths"]["log_dir"])

    preprocessed_path = "data/interim/preprocessed_telco_churn.csv"
    processed_path = config["data"]["processed_path"]

    if not os.path.exists(preprocessed_path):
        logger.error(f"Preprocessed data not found at: {preprocessed_path}")
        raise FileNotFoundError(f"Preprocessed data not found at: {preprocessed_path}")

    logger.info(f"Loading preprocessed data from {preprocessed_path}")
    df = pd.read_csv(preprocessed_path)
    logger.info(f"Input shape: {df.shape}")

    # --- Collapse 'No internet service' dummies into a single flag ---
    no_internet_cols = [col for col in df.columns if "No internet service" in col]

    if no_internet_cols:
        df["No_internet_service"] = df[no_internet_cols].any(axis=1).astype(int)
        df = df.drop(columns=no_internet_cols)
        logger.info(
            f"Collapsed {len(no_internet_cols)} 'No internet service' columns "
            f"into 'No_internet_service'"
        )
    else:
        logger.warning("No 'No internet service' columns found to collapse.")

    # --- Collapse 'MultipleLines_No phone service' into 'No_phone_service' ---
    no_phone_col = "MultipleLines_No phone service"
    if no_phone_col in df.columns:
        df["No_phone_service"] = df[no_phone_col].astype(int)
        df = df.drop(columns=[no_phone_col])
        logger.info(f"Renamed '{no_phone_col}' to 'No_phone_service'")
    else:
        logger.warning(f"Column '{no_phone_col}' not found to collapse.")

    logger.info(f"Output shape: {df.shape}")
    logger.info(f"Final columns: {list(df.columns)}")

    # --- Save ---
    os.makedirs(os.path.dirname(processed_path), exist_ok=True)
    df.to_csv(processed_path, index=False)
    logger.info(f"Processed data saved to {processed_path}")

    return df


if __name__ == "__main__":
    engineer_features()