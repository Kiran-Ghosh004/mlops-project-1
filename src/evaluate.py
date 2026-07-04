"""
Evaluation Stage
----------------
Loads the trained model and test data, produces a full classification
report and confusion matrix, and logs them as MLflow artifacts.
This is the fifth stage in the DVC pipeline.
"""
import os
import json

import joblib
import matplotlib.pyplot as plt
import mlflow
import pandas as pd
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

from utils import load_config, get_logger


def evaluate_model(config_path: str = "configs/config.yaml"):
    config = load_config(config_path)
    logger = get_logger("evaluate", config["paths"]["log_dir"])

    processed_path = config["data"]["processed_path"]
    target_column = config["data"]["target_column"]
    test_size = config["train"]["test_size"]
    random_state = config["train"]["random_state"]
    threshold = config["train"]["threshold"]
    model_dir = config["paths"]["model_dir"]

    model_path = os.path.join(model_dir, "xgboost_churn_model.pkl")
    if not os.path.exists(model_path):
        logger.error(f"Trained model not found at: {model_path}. Run train.py first.")
        raise FileNotFoundError(f"Trained model not found at: {model_path}")

    logger.info(f"Loading model from {model_path}")
    model = joblib.load(model_path)

    logger.info(f"Loading processed data from {processed_path}")
    df = pd.read_csv(processed_path)
    X = df.drop(columns=[target_column])
    y = df[target_column]

    # Same split logic as train.py — must match exactly so we evaluate
    # on the same held-out test set the model never saw during training
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    proba = model.predict_proba(X_test)[:, 1]
    y_pred = (proba >= threshold).astype(int)

    # --- Classification report ---
    report_dict = classification_report(y_test, y_pred, digits=3, output_dict=True)
    report_text = classification_report(y_test, y_pred, digits=3)
    logger.info(f"Classification report:\n{report_text}")

    # --- Confusion matrix plot ---
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["No Churn", "Churn"], yticklabels=["No Churn", "Churn"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix (threshold={threshold})")

    os.makedirs("logs", exist_ok=True)
    cm_path = "logs/confusion_matrix.png"
    fig.savefig(cm_path, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Confusion matrix saved to {cm_path}")

    # --- Save report as JSON too, for the DVC metrics file ---
    report_path = "logs/evaluation_report.json"
    with open(report_path, "w") as f:
        json.dump(report_dict, f, indent=2)
    logger.info(f"Evaluation report saved to {report_path}")

    # --- Log to MLflow (attaches to the most recent active/last run if one exists,
    # otherwise logs under a new run) ---
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("telecom-churn-xgboost")
    with mlflow.start_run(run_name="evaluation", nested=False):
        mlflow.log_metric("eval_recall_churn", report_dict["1"]["recall"])
        mlflow.log_metric("eval_precision_churn", report_dict["1"]["precision"])
        mlflow.log_metric("eval_f1_churn", report_dict["1"]["f1-score"])
        mlflow.log_metric("eval_accuracy", report_dict["accuracy"])
        mlflow.log_artifact(cm_path)
        mlflow.log_artifact(report_path)

    logger.info("Evaluation complete.")
    return report_dict


if __name__ == "__main__":
    evaluate_model()