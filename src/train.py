"""
train_model.py

Trains an ML model for heart disease prediction and saves the full
preprocessing + model pipeline to models/model_v{version}.pkl.

Structure:
- Config section
- Load data
- Preprocessing pipeline
- Model training
- Evaluation
- Save model
"""

import hashlib
import logging
import json
import sys
import time
from datetime import datetime

import pandas as pd
import joblib
import yaml
import sklearn
from pathlib import Path

from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
)
from sklearn.svm import SVC


from utils import (
    load_data,
    split_features_target,
    build_preprocessing_pipeline,
)

# Model version
MODEL_VERSION = "1.0.0"


# ---------------------------------------------------------
# Logging Setup
# ---------------------------------------------------------
class ConditionalFormatter(logging.Formatter):
    """Custom formatter that omits level name for INFO messages."""
    def format(self, record):
        if record.levelno == logging.INFO:
            return record.getMessage()
        else:
            return f"{record.levelname:<8} | {record.getMessage()}"


def setup_logging(log_dir: Path = None) -> logging.Logger:
    """Configure logging to output to both console and timestamped log file."""
    if log_dir is None:
        log_dir = Path(__file__).parent.parent / "outputs" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"training_v{MODEL_VERSION}_{timestamp}.log"

    logger = logging.getLogger("heart_disease_prediction_training")
    logger.setLevel(logging.DEBUG)
    logger.handlers = []

    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(ConditionalFormatter())

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(ConditionalFormatter())

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    logger.info(f"Logging initialized. Log file: {log_file}")
    return logger


logger = setup_logging()


# ---------------------------------------------------------
# Config
# ---------------------------------------------------------
PROJECT_ROOT = Path(__file__).parent.parent
CONFIG_DIR = PROJECT_ROOT / "config"

try:
    with open(CONFIG_DIR / "train_config.yaml") as f:
        train_cfg = yaml.safe_load(f)
    with open(CONFIG_DIR / "model_config.yaml") as f:
        model_cfg = yaml.safe_load(f)
except FileNotFoundError as e:
    logger.error(f"Config file not found: {e}")
    raise
except yaml.YAMLError as e:
    logger.error(f"Error parsing YAML config: {e}")
    raise

DATA_PATH = PROJECT_ROOT / train_cfg["data"]["input_path"]
MODEL_PATH = PROJECT_ROOT / train_cfg["output"]["model_path"]
TEST_SIZE = train_cfg["training"]["test_size"]
RANDOM_STATE = train_cfg["training"]["random_state"]
TARGET_COL = train_cfg["data"]["target_col"]
MODEL_PARAMS = model_cfg["model"]["params"]


# ---------------------------------------------------------
# Metadata
# ---------------------------------------------------------
def get_data_hash(filepath: Path) -> str:
    """Generate MD5 hash of data file for tracking data versions."""
    try:
        with open(filepath, "rb") as f:
            return hashlib.md5(f.read()).hexdigest()[:12]
    except Exception:
        return "unknown"


def build_metadata(metrics: dict) -> dict:
    """Build metadata dict for model artifact."""
    return {
        "model_version": MODEL_VERSION,
        "trained_at": datetime.now().isoformat(),
        "data_hash": get_data_hash(DATA_PATH),
        "metrics": metrics,
        "model_params": MODEL_PARAMS,
        "python_version": (
            f"{sys.version_info.major}."
            f"{sys.version_info.minor}."
            f"{sys.version_info.micro}"
        ),
        "sklearn_version": sklearn.__version__,
        "pandas_version": pd.__version__,
    }


# ---------------------------------------------------------
# Training Pipeline
# ---------------------------------------------------------
def build_model(model_params: dict, random_state: int = RANDOM_STATE) -> SVC:
    """Build an SVC with the given parameters."""
    model = SVC(
        random_state=random_state,
        **model_params
    )
    return model


def build_training_pipeline(
    preprocessor: ColumnTransformer,
    model_params: dict,
    random_state: int = RANDOM_STATE
) -> Pipeline:
    """Build a complete training pipeline with preprocessing and model."""
    pipeline = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            (
                "model",
                build_model(
                    model_params=model_params, random_state=random_state
                ),
            ),
        ]
    )
    return pipeline


# ---------------------------------------------------------
# Train + Evaluate
# ---------------------------------------------------------
def log_environment_info() -> None:
    """Log environment and version information for reproducibility."""
    logger.info("=" * 60)
    logger.info("ENVIRONMENT INFO")
    logger.info("=" * 60)
    logger.info(f"Python version: {sys.version}")
    logger.info(f"Scikit-learn version: {sklearn.__version__}")
    logger.info(f"Pandas version: {pd.__version__}")
    logger.info(f"Working directory: {Path.cwd()}")
    logger.info(f"Script location: {Path(__file__).resolve()}")


def log_training_config(
    model_params: dict,
    test_size: float,
    random_state: int,
) -> None:
    """Log training configuration and hyperparameters."""
    logger.info("=" * 60)
    logger.info("TRAINING CONFIGURATION")
    logger.info("=" * 60)
    logger.info(f"Test size: {test_size}")
    logger.info(f"Random state: {random_state}")
    logger.info(f"Model parameters: {json.dumps(model_params, indent=2)}")


def log_data_info(X_train, X_test, y_train, y_test) -> None:
    """Log dataset information."""
    logger.info("=" * 60)
    logger.info("DATA INFO")
    logger.info("=" * 60)
    logger.info(f"Training samples: {len(X_train)}")
    logger.info(f"Test samples: {len(X_test)}")
    logger.info(f"Features: {X_train.shape[1]}")
    logger.info(f"Feature names: {list(X_train.columns)}")
    logger.info(f"Target distribution (train): {dict(y_train.value_counts())}")
    logger.info(f"Target distribution (test): {dict(y_test.value_counts())}")


def log_metrics(y_test, y_pred, y_proba, training_time: float) -> dict:
    """Log comprehensive evaluation metrics."""
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred),
        "training_time_seconds": round(training_time, 2),
    }

    logger.info("=" * 60)
    logger.info("EVALUATION METRICS")
    logger.info("=" * 60)
    for metric_name, value in metrics.items():
        if metric_name == "training_time_seconds":
            logger.info(f"{metric_name}: {value}s")
        else:
            logger.info(f"{metric_name}: {value:.4f}")

    logger.info("-" * 60)
    logger.info("Classification Report:")
    logger.info("\n" + classification_report(y_test, y_pred))

    return metrics


def train_and_evaluate(
    df: pd.DataFrame,
    model_params: dict,
    target_col: str = TARGET_COL,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> Pipeline:
    """Train and evaluate a model pipeline."""
    log_training_config(model_params, test_size, random_state)

    X, y = split_features_target(df, target_col=target_col)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    log_data_info(X_train, X_test, y_train, y_test)

    preprocessor = build_preprocessing_pipeline(X)
    pipeline = build_training_pipeline(
        preprocessor,
        model_params=model_params,
        random_state=random_state,
    )

    logger.info("=" * 60)
    logger.info("TRAINING")
    logger.info("=" * 60)
    logger.info("Starting model training...")

    start_time = time.time()
    pipeline.fit(X_train, y_train)
    training_time = time.time() - start_time

    logger.info(f"Training completed in {training_time:.2f} seconds")

    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    metrics = log_metrics(y_test, y_pred, y_proba, training_time)

    return pipeline, metrics


# ---------------------------------------------------------
# Save Model
# ---------------------------------------------------------
def save_model(pipeline: Pipeline, metrics: dict, model_dir: Path) -> Path:
    """Save the trained pipeline with metadata and create latest symlink."""
    model_dir.mkdir(parents=True, exist_ok=True)
    filename = f"model_v{MODEL_VERSION}.pkl"

    metadata = build_metadata(metrics)
    artifact = {
        "pipeline": pipeline,
        "metadata": metadata,
    }

    model_path = model_dir / filename
    joblib.dump(artifact, model_path)
    logger.info(f"Saved model to {model_path}")

    logger.info("Embedded metadata:")
    logger.info(f"  - model_version: {metadata['model_version']}")
    logger.info(f"  - data_hash: {metadata['data_hash']}")
    logger.info(f"  - sklearn_version: {metadata['sklearn_version']}")

    latest_link = model_dir / "latest.pkl"
    if latest_link.is_symlink() or latest_link.exists():
        latest_link.unlink()
    latest_link.symlink_to(filename)
    logger.info(f"Updated symlink: {latest_link} -> {filename}")

    return model_path


# ---------------------------------------------------------
# Main Script Entry Point
# ---------------------------------------------------------
def main():
    """Main training pipeline with comprehensive logging."""
    logger.info("=" * 60)
    logger.info("HEART DISEASE PREDICTION - MODEL TRAINING")
    logger.info(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 60)

    log_environment_info()
    try:
        logger.info("=" * 60)
        logger.info("LOADING DATA")
        logger.info("=" * 60)
        logger.info(f"Data path: {DATA_PATH}")
        df = load_data(DATA_PATH)
        logger.info(f"Loaded {len(df)} records with {len(df.columns)} columns")

        pipeline, metrics = train_and_evaluate(df, model_params=MODEL_PARAMS)
        save_model(pipeline, metrics, MODEL_PATH.parent)

        logger.info("=" * 60)
        logger.info("TRAINING COMPLETED SUCCESSFULLY")
        finished_at = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        logger.info(f"Finished at: {finished_at}")
        logger.info("=" * 60)

    except Exception as e:
        logger.error(f"Training failed with error: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
