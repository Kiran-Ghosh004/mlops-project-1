"""Shared helpers: config loading and logging setup."""
import os
import logging
import yaml


def load_config(config_path: str = "configs/config.yaml") -> dict:
    """Load the YAML config file into a dict."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found at: {config_path}")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    return config


def get_logger(name: str, log_dir: str = "logs/") -> logging.Logger:
    """Create a logger that writes to both console and a log file."""
    os.makedirs(log_dir, exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        formatter = logging.Formatter(
            "%(asctime)s | %(name)s | %(levelname)s | %(message)s"
        )

        file_handler = logging.FileHandler(os.path.join(log_dir, f"{name}.log"))
        file_handler.setFormatter(formatter)

        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)

        logger.addHandler(file_handler)
        logger.addHandler(stream_handler)

    return logger