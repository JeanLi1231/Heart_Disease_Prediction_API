# Heart Disease Prediction API

Production-style FastAPI ML inference service for heart disease risk prediction, with Docker, CI, and reproducible training artifacts.

This project demonstrates how to take a trained ML model and expose it as a robust, testable HTTP service, following common industry patterns.


## Features

- Reproducible ML pipeline: data loading, preprocessing, training, evaluation, and artifact versioning
- Versioned model metadata exposed via API for traceability
- REST API built with FastAPI
- Dockerized for consistent local and CI execution
- CI workflow with linting, API contract tests (FastAPI TestClient), and live container tests
- Configuration driven by YAML files


## Project Structure
```
.
├── app/                    # FastAPI application
│   └── main.py
├── src/                    # Training and inference logic
│   ├── train.py
│   ├── inference.py
│   └── utils/
│       └── data.py
├── notebooks/              # Exploration and parameter tuning
│   ├── exploration.ipynb
│   └── tuning.ipynb
├── config/                 # YAML configuration files
│   ├── train_config.yaml
│   └── model_config.yaml
├── tests/                  # API contract and live container tests
│   ├── test_api.py
│   ├── test_api_live.py
│   └── test_api_live_withmodel.py
├── Dockerfile
├── requirements.txt
├── requirements-dev.txt
└── README.md
```


## API Endpoints

The API exposes a small, production-style interface for model inference and service health monitoring.


### Health Check (Unversioned)

`GET /health`

Used for container orchestration, Docker health checks, and uptime monitoring.
This endpoint is intentionally unversioned so infrastructure does not need to change when API versions evolve.

Response
```json
   {
     "status": "ok"
   }   
```


### Predict (Versioned)

`POST /v1/predict`

Runs inference on a single patient record.

Request Body
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

Successful Response
```json
   {
    "prediction": 0,
    "probability": 0.09320092545819073,
    "diagnosis": "No Heart Disease"
   }   
```

Error Responses
- 422 — validation error (missing or invalid fields)
- 503 — model artifact not loaded


### Model Metadata (Versioned)

`GET /v1/model-info`

Returns metadata embedded at training time.

Response
```json
    {
      "model_version": "1.0.0",
      "trained_at": "2026-01-17T13:59:04.269416",
      "data_hash": "ab21f2524241",
      "metrics": {
        "accuracy": 0.8478260869565217,
        "roc_auc": 0.9179818268770923,
        "precision": 0.87,
        "recall": 0.8529411764705882,
        "f1_score": 0.8613861386138614,
        "training_time_seconds": 0.02
      },
      "model_params": {
        "C": 1,
        "class_weight": null,
        "gamma": "auto",
        "kernel": "rbf",
        "probability": true
      },
      "python_version": "3.11.14",
      "sklearn_version": "1.7.2",
      "pandas_version": "2.3.3"
    }
```

If the model artifact is not present, this endpoint returns:
```json   
    {}   
```


## Model Artifact Design

The trained model is saved as a single artifact containing:

- pipeline: full sklearn preprocessing plus estimator pipeline
- metadata: model version, training timestamp, data hash, evaluation metrics, model params, python, sklearn and pandas versions

This ensures:
- Training and inference stay consistent
- Metadata travels with the model
- No duplicated configuration in the API

The API lazy-loads the model at runtime to avoid import-time failures in CI and testing environments.


## Docker Usage

Build
```bash
   docker build -t heart-disease-api:1.0 .
```

Run
```bash
   docker run -p 8000:8000 heart-disease-api:1.0
```

Health Check
```bash
   curl http://localhost:8000/health
```


## Testing Strategy

This project follows a layered testing approach.

Integration / API Contract Tests
- Located in tests/test_api.py
- Use FastAPI TestClient
- Inject a dummy sklearn pipeline
- Do not require a real model.pkl
- Fast and deterministic
- Run on every CI push

Live / Container Integration Tests (No Model)
- Located in tests/test_api_live.py
- Validate container startup and routing
- Expect 503 from prediction endpoints
- Used in CI to ensure infrastructure correctness

Live / Container Integration Tests (With Model)
- Located in tests/test_api_live_withmodel.py
- Run against a container with a real model artifact
- Validate full inference path
- Intended for manual runs or release pipelines


## API Versioning Strategy

- /health is unversioned
- All inference endpoints are versioned (/v1/...) using FastAPI routers for version control
- New versions (/v2) can reuse the same logic while swapping models or pipelines
- Infrastructure and Docker configuration remain unchanged


## Scope and Non-Goals

This project intentionally focuses on:
- Clean ML to API to Docker integration
- Reproducible training artifacts
- CI-safe testing patterns

Out of scope:
- Authentication and authorization
- Feature stores
- Autoscaling and cloud deployment
- Model monitoring and drift detection
