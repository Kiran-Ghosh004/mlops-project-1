"""
Prediction / Inference Logic
------------------------------
Shared inference logic used by the FastAPI service (and optionally batch
scripts). Takes raw customer data (same schema as the original CSV, minus
the target column), applies the same encoding + feature engineering used
during training, aligns columns to match the trained model exactly, and
returns a churn prediction + probability.
"""
import json
import os

import joblib
import pandas as pd

pd.set_option('future.no_silent_downcasting', True)

from utils import load_config, get_logger

BINARY_COLS = ["gender", "Partner", "Dependents", "PhoneService", "PaperlessBilling"]

MULTI_CAT_COLS = [
    "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
    "Contract", "PaymentMethod"
]


class ChurnPredictor:
    """Loads the trained model once and serves predictions on raw input data."""

    def __init__(self, config_path: str = "configs/config.yaml"):
        self.config = load_config(config_path)
        self.logger = get_logger("predict", self.config["paths"]["log_dir"])

        model_dir = self.config["paths"]["model_dir"]
        model_path = os.path.join(model_dir, "xgboost_churn_model.pkl")
        feature_columns_path = os.path.join(model_dir, "feature_columns.json")

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at: {model_path}. Run train.py first.")
        if not os.path.exists(feature_columns_path):
            raise FileNotFoundError(f"Feature columns file not found at: {feature_columns_path}")

        self.model = joblib.load(model_path)
        with open(feature_columns_path, "r") as f:
            self.feature_columns = json.load(f)

        self.threshold = self.config["train"]["threshold"]
        self.logger.info("ChurnPredictor initialized: model and feature columns loaded.")

    def _preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """Replicates data_preprocessing.py + feature_engineering.py for raw input."""
        df = df.copy()

        # --- Binary encoding ---
        df[BINARY_COLS] = df[BINARY_COLS].replace({
            "Yes": 1, "No": 0,
            "Male": 1, "Female": 0
        })

        # --- One-hot encoding ---
        df = pd.get_dummies(df, columns=MULTI_CAT_COLS, drop_first=True)

        # --- Clean TotalCharges ---
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
        df["TotalCharges"] = df["TotalCharges"].fillna(0)

        # --- Drop customerID if present ---
        if "customerID" in df.columns:
            df = df.drop(columns=["customerID"])

        # --- Convert bool columns to int ---
        bool_cols = df.select_dtypes(include="bool").columns
        df[bool_cols] = df[bool_cols].astype(int)

        # --- Collapse 'No internet service' dummies ---
        no_internet_cols = [col for col in df.columns if "No internet service" in col]
        if no_internet_cols:
            df["No_internet_service"] = df[no_internet_cols].any(axis=1).astype(int)
            df = df.drop(columns=no_internet_cols)

        # --- Collapse 'MultipleLines_No phone service' ---
        no_phone_col = "MultipleLines_No phone service"
        if no_phone_col in df.columns:
            df["No_phone_service"] = df[no_phone_col].astype(int)
            df = df.drop(columns=[no_phone_col])

        # --- Align columns to match training exactly ---
        # Add any missing columns (e.g. a category that didn't appear in this
        # particular input) as 0, and drop any extras, then reorder to match
        # the exact column order the model was trained on.
        for col in self.feature_columns:
            if col not in df.columns:
                df[col] = 0

        df = df[self.feature_columns]

        return df

    def predict(self, raw_data: dict) -> dict:
        """
        Takes a single customer record as a dict (raw schema, matching the
        original CSV columns minus 'Churn'), returns a prediction.
        """
        df = pd.DataFrame([raw_data])
        processed_df = self._preprocess(df)

        proba = self.model.predict_proba(processed_df)[:, 1][0]
        prediction = int(proba >= self.threshold)

        result = {
            "churn_prediction": prediction,
            "churn_probability": round(float(proba), 4),
            "threshold_used": self.threshold,
        }
        self.logger.info(f"Prediction made: {result}")
        return result


if __name__ == "__main__":
    # Quick manual test with a sample customer record
    sample_customer = {
        "customerID": "0000-TEST",
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 1,
        "PhoneService": "No",
        "MultipleLines": "No phone service",
        "InternetService": "DSL",
        "OnlineSecurity": "No",
        "OnlineBackup": "Yes",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 29.85,
        "TotalCharges": 29.85,
    }

    predictor = ChurnPredictor()
    result = predictor.predict(sample_customer)
    print(result)