# Heart Disease Prediction API

A production-grade machine learning project for predicting heart disease using clinical data. This repository includes data processing, model training, evaluation, and a FastAPI-based REST API for real-time predictions.

## Features
- End-to-end ML pipeline from data ingestion to a containerized inference service (training, preprocessing, evaluation, and versioned artifacts)
- Model metadata and versioning for reproducibility and traceability
- REST API built with FastAPI for health checks, predictions, and model metadata
- Dockerized application for reproducible, deployment-ready inference
- CI workflow with linting, API integration tests, and live Docker-based tests
- Configuration-driven training and inference via YAML files

## Project Structure
```
├── app/                # FastAPI app and model artifact
├── config/             # YAML config files for training and model
├── data/               # Input data (CSV)
├── models/             # Saved model artifacts (versioned)
├── notebooks/          # Jupyter notebooks for exploration and parameter tuning
├── outputs/            # Predictions and logs
├── src/                # Source code (training, inference, utils)
├── tests/              # Integration and live API tests
├── Dockerfile          # Containerization
├── requirements.txt    # Runtime dependencies
├── requirements-dev.txt# Dev/test dependencies
└── README.md           # Project documentation
```

## Getting Started

### 1. Clone the repository
```sh
git clone <repo-url>
cd Heart_Disease_Prediction
```

### 2. Set up the environment
```sh
python -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
# Optional: install dev/test/jupyter dependencies used for development and CI
# (not required in production/runtime images)
pip install -r requirements-dev.txt  # optional
```

### 3. Train the model
Edit config files in `config/` as needed, then run:
```sh
python src/train.py
```
Model artifact will be saved in `models/`.

### 4. Run the API
```sh
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### 5. Make predictions
- Health check: `GET /health`
- Predict: `POST /predict` (JSON body with patient features)
- Model info: `GET /model-info`

Example prediction request:
```json
{
  "Age": 54,
  "Sex": "M",
  "ChestPainType": "ASY",
  "RestingBP": 150,
  "Cholesterol": 195,
  "FastingBS": 0,
  "RestingECG": "Normal",
  "MaxHR": 122,
  "ExerciseAngina": "N",
  "Oldpeak": 0,
  "ST_Slope": "Up"
}
```

### 6. Run tests
```sh
pytest -v tests/
```

### 7. Build and run with Docker
```sh
docker build -t heart-disease-prediction-api .
docker run -p 8000:8000 heart-disease-prediction-api
```

## CI/CD
- GitHub Actions workflow: Linting, integration tests, Docker build, and live API tests

## Configuration
- `config/train_config.yaml`: Data paths, training/test split, output locations
- `config/model_config.yaml`: Model hyperparameters
