"""
Hyperparameter Tuning Stage (Optuna)
--------------------------------------
Runs an Optuna study to find the best XGBoost hyperparameters,
optimizing for recall on the churn class at a fixed decision threshold.
This mirrors the tuning done in the EDA notebook.

This stage is run manually/occasionally (not on every DVC pipeline run),
since re-tuning every time would be slow and the best params are already
saved in configs/config.yaml for the standard train.py stage to use.
"""
import json
import os

import optuna
import pandas as pd
from sklearn.metrics import recall_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from utils import load_config, get_logger


def run_tuning(config_path: str = "configs/config.yaml", n_trials: int = 30):
    config = load_config(config_path)
    logger = get_logger("tune", config["paths"]["log_dir"])

    processed_path = config["data"]["processed_path"]
    target_column = config["data"]["target_column"]
    test_size = config["train"]["test_size"]
    random_state = config["train"]["random_state"]
    threshold = config["train"]["threshold"]

    logger.info(f"Loading processed data from {processed_path}")
    df = pd.read_csv(processed_path)

    X = df.drop(columns=[target_column])
    y = df[target_column]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    logger.info(f"scale_pos_weight computed: {scale_pos_weight:.4f}")

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 300, 800),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "gamma": trial.suggest_float("gamma", 0, 5),
            "reg_alpha": trial.suggest_float("reg_alpha", 0, 5),
            "reg_lambda": trial.suggest_float("reg_lambda", 0, 5),
            "random_state": random_state,
            "n_jobs": -1,
            "scale_pos_weight": scale_pos_weight,
            "eval_metric": "logloss",
        }

        model = XGBClassifier(**params)
        model.fit(X_train, y_train)
        proba = model.predict_proba(X_test)[:, 1]
        y_pred = (proba >= threshold).astype(int)
        return recall_score(y_test, y_pred, pos_label=1)

    logger.info(f"Starting Optuna study: {n_trials} trials, maximizing recall")
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)

    logger.info(f"Best recall: {study.best_value:.4f}")
    logger.info(f"Best params: {study.best_params}")

    # Save best params to a JSON file for reference (config.yaml stays the source
    # of truth for train.py, but this gives you a record of each tuning run)
    output_path = "configs/tuning_results.json"
    with open(output_path, "w") as f:
        json.dump({
            "best_recall": study.best_value,
            "best_params": study.best_params,
            "scale_pos_weight": scale_pos_weight,
            "threshold": threshold,
            "n_trials": n_trials,
        }, f, indent=2)
    logger.info(f"Tuning results saved to {output_path}")

    return study


if __name__ == "__main__":
    run_tuning()