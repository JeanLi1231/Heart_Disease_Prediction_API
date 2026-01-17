"""
main.py

FastAPI application for heart disease prediction.

Load trained sklearn pipeline artifact
Validate input schema
Run inference only (no training)
Return prediction + probability
"""

from pathlib import Path
import logging

import pandas as pd
import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


# ---------------------------------------------------------
# App & Logging
# ---------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("heart_disease_prediction_api")

app = FastAPI(
    title="Heart Disease Prediction API",
    description="Predicts risk of heart disease using a trained ML pipeline."
)


# ---------------------------------------------------------
# Model artifact location + lazy-loaded globals
# ---------------------------------------------------------
APP_ROOT = Path(__file__).parent
MODEL_PATH = APP_ROOT / "model.pkl"

_pipeline = None
_model_metadata = {}
_load_attempted = False


def _load_artifact():
    global _pipeline, _model_metadata, _load_attempted

    if _load_attempted:
        return
    _load_attempted = True

    try:
        artifact = joblib.load(MODEL_PATH)
        _pipeline = artifact.get("pipeline")
        _model_metadata = artifact.get("metadata", {})
        logger.info(
            f"Loaded model v{_model_metadata.get('model_version', 'unknown')} "
            f"from {MODEL_PATH}"
        )
    except FileNotFoundError:
        logger.warning(
            f"Model artifact not found at {MODEL_PATH}. "
            "Prediction endpoints will return 503 until a model is provided."
        )
        _pipeline = None
        _model_metadata = {}
    except Exception as e:
        logger.error(
            f"Model artifact failed to load from {MODEL_PATH}: {e}",
            exc_info=True
        )
        _pipeline = None
        _model_metadata = {}


def get_pipeline():
    """Public accessor for the pipeline. Ensures artifact is loaded."""
    _load_artifact()
    return _pipeline


def get_model_metadata():
    """Public accessor for model metadata. Ensures artifact is loaded."""
    _load_artifact()
    return _model_metadata


# ---------------------------------------------------------
# Input Schema
# ---------------------------------------------------------
class PatientFeatures(BaseModel):
    Age: int = Field(...)
    Sex: str = Field(...)
    ChestPainType: str = Field(...)
    RestingBP: int = Field(...)
    Cholesterol: int = Field(...)
    FastingBS: int = Field(...)
    RestingECG: str = Field(...)
    MaxHR: int = Field(...)
    ExerciseAngina: str = Field(...)
    Oldpeak: float = Field(...)
    ST_Slope: str = Field(...)

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "Age": 40,
                    "Sex": "M",
                    "ChestPainType": "ATA",
                    "RestingBP": 140,
                    "Cholesterol": 289,
                    "FastingBS": 0,
                    "RestingECG": "Normal",
                    "MaxHR": 172,
                    "ExerciseAngina": "N",
                    "Oldpeak": 0,
                    "ST_Slope": "Up"
                }
            ]
        }
    }


# ---------------------------------------------------------
# Response Schemas
# ---------------------------------------------------------
class PredictionResponse(BaseModel):
    prediction: int = Field(
        ...,
        description=(
            "0 = No heart disease, 1 = Heart disease"
        ),
    )

    probability: float = Field(
        ...,
        description=(
            "Probability of heart disease (0-1)"
        ),
    )


class HealthResponse(BaseModel):
    status: str = Field(...)

    model_config = {
        "json_schema_extra": {
            "examples": [
                {"status": "ok"}
            ]
        }
    }


# ---------------------------------------------------------
# Health Check
# ---------------------------------------------------------
@app.get("/health", response_model=HealthResponse)
def health_check():
    return HealthResponse(status="ok")


# ---------------------------------------------------------
# Prediction Endpoint
# ---------------------------------------------------------
@app.post("/predict", response_model=PredictionResponse)
def predict(features: PatientFeatures):
    """
    Run inference on a single patient record.
    """
    pipeline = get_pipeline()
    if pipeline is None:
        logger.warning("Prediction requested but model is not loaded")
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        input_df = pd.DataFrame([features.model_dump()])
        prediction = pipeline.predict(input_df)[0]
        proba = pipeline.predict_proba(input_df)[0]

        # Determine probability for positive class (1).
        # Handle classifiers that were trained with only a single class.
        try:
            final_est = pipeline.steps[-1][1]
        except Exception:
            final_est = None

        if final_est is not None:
            classes = getattr(final_est, "classes_", None)
            if classes is not None:
                if 1 in list(classes):
                    idx = list(classes).index(1)
                    probability = float(proba[idx])
                else:
                    probability = 0.0
            else:
                probability = (
                    float(proba[1]) if proba.shape[0] > 1 else float(proba[0])
                )
        else:
            probability = (
                float(proba[1]) if proba.shape[0] > 1 else float(proba[0])
            )

        return {
            "prediction": int(prediction),
            "probability": float(probability)
        }

    except KeyError as e:
        logger.error(f"Missing feature: {e}", exc_info=True)
        raise HTTPException(status_code=400, detail=f"Missing feature: {e}")
    except ValueError as e:
        logger.error(f"Invalid input values: {e}", exc_info=True)
        raise HTTPException(status_code=400, detail="Invalid input values")
    except Exception as e:
        logger.error(f"Inference failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Prediction failed")


# ---------------------------------------------------------
# Model Info Endpoint
# ---------------------------------------------------------
@app.get("/model-info")
def model_info():
    """Return model metadata (version, training time, metrics, etc)."""
    return get_model_metadata()
