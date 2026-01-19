"""inference.py

Load the trained model and make predictions on new data.

Usage:
  - From command line with CSV file:
      python src/inference.py --input data/new_patients.csv \
          --output outputs/predictions.csv

  - From command line with single sample (JSON):
      python src/inference.py --sample '{
      "Age": 54, "Sex": "M", "ChestPainType": "ASY", "RestingBP": 150,
      "Cholesterol": 195, "FastingBS": 0, "RestingECG": "Normal",
      "MaxHR": 122, "ExerciseAngina": "N", "Oldpeak": 0, "ST_Slope": "Up"}'

  - As a module:
      from src import load_model, predict
      model = load_model()
      predictions = predict(model, df)
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import joblib
import pandas as pd
from sklearn.pipeline import Pipeline

from utils.data import load_data


# ---------------------------------------------------------
# Logging Setup
# ---------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------
# Config
# ---------------------------------------------------------
PROJECT_ROOT = Path(__file__).parent.parent
MODEL_DIR = PROJECT_ROOT / "models"
DEFAULT_MODEL_PATH = MODEL_DIR / "latest.pkl"


# ---------------------------------------------------------
# Model Loading
# ---------------------------------------------------------
def load_model(model_path: Path = DEFAULT_MODEL_PATH) -> Pipeline:
    """Load a trained model pipeline from disk.

    Args:
        model_path: Path to the model file. Defaults to latest.pkl.

    Returns:
        Trained sklearn Pipeline with preprocessing and model.
    """
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found at {model_path}."
            " Please run training first: python src/train.py"
        )

    logger.info(f"Loading model from: {model_path}")

    if model_path.is_symlink():
        actual_path = model_path.resolve()
        logger.info(
            f"Resolved symlink to: {actual_path.name}"
        )

    artifact = joblib.load(model_path)

    pipeline = artifact["pipeline"]
    metadata = artifact.get("metadata", {})
    logger.info(
        f"Model version: {metadata.get('model_version', 'unknown')}"
    )
    logger.info(
        f"Trained at: {metadata.get('trained_at', 'unknown')}"
    )

    logger.info("Model loaded successfully")
    return pipeline


# ---------------------------------------------------------
# Prediction
# ---------------------------------------------------------
def predict(model, X: pd.DataFrame) -> pd.DataFrame:
    """Make predictions on new data.

    Args:
      model: Trained sklearn Pipeline.
      X: DataFrame with features (same columns as training data,
         excluding target).

    Returns:
      DataFrame with predictions and probabilities.
    """
    logger.info(f"Making predictions for {len(X)} samples...")

    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)[:, 1]

    results = X.copy()
    results["prediction"] = y_pred
    results["probability"] = y_proba

    logger.info(
        f"Predictions complete: {sum(y_pred)} positive, "
        f"{len(y_pred) - sum(y_pred)} negative"
    )

    return results


def predict_single(model, sample: dict) -> dict:
    """Make prediction for a single sample.

    Args:
      model: Trained sklearn Pipeline.
      sample: Dictionary with feature values.

    Returns:
      Dictionary with prediction results.
    """
    df = pd.DataFrame([sample])
    results = predict(model, df)

    pred = int(results["prediction"].iloc[0])
    proba = float(results["probability"].iloc[0])
    diagnosis = (
        "Heart Disease Detected" if pred == 1 else "No Heart Disease"
    )

    return {
        "prediction": pred,
        "probability": proba,
        "diagnosis": diagnosis,
    }


# ---------------------------------------------------------
# CLI Interface
# ---------------------------------------------------------
def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Make heart disease predictions using trained model."
    )
    parser.add_argument(
        "--model",
        type=str,
        default=str(DEFAULT_MODEL_PATH),
        help="Path to model file (default: models/latest.pkl)"
    )
    parser.add_argument(
        "--input",
        type=str,
        help="Path to input CSV file with patient data"
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Path to save predictions CSV (default: print to stdout)"
    )
    parser.add_argument(
        "--sample",
        type=str,
        help="Single sample as JSON string for prediction"
    )
    return parser.parse_args()


def main():
    """Main entry point for CLI."""
    args = parse_args()

    model_path = Path(args.model)
    model = load_model(model_path)

    if args.sample:
        try:
            sample = json.loads(args.sample)
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON: {e}")
            sys.exit(1)

        result = predict_single(model, sample)
        print("\n" + "=" * 40)
        print("PREDICTION RESULT")
        print("=" * 40)
        for key, value in result.items():
            print(f"{key}: {value}")
        print("=" * 40)
        return

    if args.input:
        input_path = Path(args.input)
        if not input_path.exists():
            logger.error(f"Input file not found: {input_path}")
            sys.exit(1)

        df = load_data(input_path)

        if "HeartDisease" in df.columns:
            df = df.drop("HeartDisease", axis=1)

        results = predict(model, df)

        if args.output:
            output_path = Path(args.output)
            results.to_csv(output_path, index=False)
            logger.info(f"Predictions saved to: {output_path}")
        else:
            print("\n" + results.to_string())
        return

    logger.error("Please provide --input CSV file or --sample JSON")
    sys.exit(1)


if __name__ == "__main__":
    main()
