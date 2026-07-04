"""
Training Stage
--------------
Trains the final XGBoost model using the best hyperparameters found via
Optuna (stored in configs/config.yaml), logs the run to MLflow, and saves
the trained model artifact to disk.
This is the fourth stage in the DVC pipeline.
"""
import os
import json
import time

import joblib
import mlflow
import mlflow.xgboost
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from utils import load_config, get_logger


def train_model(config_path: str = "configs/config.yaml"):
    config = load_config(config_path)
    logger = get_logger("train", config["paths"]["log_dir"])

    processed_path = config["data"]["processed_path"]
    target_column = config["data"]["target_column"]
    test_size = config["train"]["test_size"]
    random_state = config["train"]["random_state"]
    threshold = config["train"]["threshold"]
    best_params = dict(config["model"]["best_params"])
    model_dir = config["paths"]["model_dir"]

    if not os.path.exists(processed_path):
        logger.error(f"Processed data not found at: {processed_path}")
        raise FileNotFoundError(f"Processed data not found at: {processed_path}")

    logger.info(f"Loading processed data from {processed_path}")
    df = pd.read_csv(processed_path)

    X = df.drop(columns=[target_column])
    y = df[target_column]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    logger.info(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")

    # scale_pos_weight depends on the actual training split, so it's
    # computed at train time rather than hardcoded in config.yaml
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    best_params["scale_pos_weight"] = scale_pos_weight
    logger.info(f"scale_pos_weight computed: {scale_pos_weight:.4f}")

    mlflow.set_tracking_uri("http://localhost:5000")

    mlflow.set_experiment("telecom-churn-xgboost")

    with mlflow.start_run(run_name="xgboost_best_params"):
        # --- Log params ---
        mlflow.log_params(best_params)
        mlflow.log_param("threshold", threshold)
        mlflow.log_param("test_size", test_size)

        # --- Train ---
        model = XGBClassifier(**best_params)

        start_train = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - start_train
        logger.info(f"Training completed in {train_time:.2f}s")
        mlflow.log_metric("train_time_seconds", train_time)

        # --- Predict (needed here only to log basic training-time metrics;
        # full evaluation with confusion matrix etc. lives in evaluate.py) ---
        start_pred = time.time()
        proba = model.predict_proba(X_test)[:, 1]
        pred_time = time.time() - start_pred
        y_pred = (proba >= threshold).astype(int)
        mlflow.log_metric("predict_time_seconds", pred_time)

        from sklearn.metrics import recall_score, precision_score, f1_score
        recall = recall_score(y_test, y_pred, pos_label=1)
        precision = precision_score(y_test, y_pred, pos_label=1)
        f1 = f1_score(y_test, y_pred, pos_label=1)

        mlflow.log_metric("recall", recall)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("f1_score", f1)
        logger.info(f"Recall: {recall:.3f} | Precision: {precision:.3f} | F1: {f1:.3f}")

        # --- Log model to MLflow ---
        mlflow.xgboost.log_model(model, "model")

        # --- Save model artifact locally too (for the FastAPI service to load) ---
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, "xgboost_churn_model.pkl")
        joblib.dump(model, model_path)
        logger.info(f"Model saved locally to {model_path}")

        # Save the exact feature column order — critical for the API later,
        # since one-hot encoded columns must match at inference time
        feature_columns_path = os.path.join(model_dir, "feature_columns.json")
        with open(feature_columns_path, "w") as f:
            json.dump(list(X.columns), f, indent=2)
        logger.info(f"Feature columns saved to {feature_columns_path}")

        run_id = mlflow.active_run().info.run_id
        logger.info(f"MLflow run ID: {run_id}")

    return model, X_test, y_test


if __name__ == "__main__":
    train_model()