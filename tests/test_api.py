from fastapi.testclient import TestClient
from sklearn.dummy import DummyClassifier
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import app.main as main_mod
from app.main import app


# Create a dummy pipeline and artifact so tests don't require app/model.pkl
class IdentityTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X


dummy_pipeline = Pipeline(
    steps=[
        ("identity", IdentityTransformer()),
        ("clf", DummyClassifier(strategy="most_frequent")),
    ]
)

dummy_artifact = {
    "pipeline": dummy_pipeline,
    "metadata": {
        "model_version": "test",
        "trained_at": "2025-01-01",
        "metrics": {},
    },
}

VALID_SAMPLE = {
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
    "ST_Slope": "Up",
}

# Fit the dummy pipeline so it can predict during tests
dummy_pipeline.fit(pd.DataFrame([VALID_SAMPLE]), [0])


def _fake_get_pipeline():
    return dummy_pipeline


def _fake_get_model_metadata():
    return dummy_artifact["metadata"]


# Patch the accessor functions (cleaner than touching private globals)
main_mod.get_pipeline = _fake_get_pipeline
main_mod.get_model_metadata = _fake_get_model_metadata


client = TestClient(app)


def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict_valid():
    response = client.post("/predict", json=VALID_SAMPLE)
    assert response.status_code == 200
    data = response.json()
    assert "prediction" in data
    assert "probability" in data
    assert isinstance(data["probability"], float)


def test_predict_missing_fields():
    response = client.post("/predict", json={"Age": 54})
    assert response.status_code == 422
    data = response.json()
    assert "detail" in data
    print("DEBUG VALIDATION ERROR:", data["detail"])
    assert any(
        "Field required" in err.get("msg", "") for err in data["detail"]
    )


def test_model_info():
    response = client.get("/model-info")
    assert response.status_code == 200
    data = response.json()
    assert "model_version" in data
    assert "trained_at" in data
    assert "metrics" in data
