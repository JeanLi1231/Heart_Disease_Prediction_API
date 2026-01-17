import time
import requests

BASE_URL = "http://localhost:8000"

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
    "ST_Slope": "Up"
}


def wait_for_api(timeout: int = 30):
    """
    Wait until the API is responsive.
    Useful in CI where container startup takes time.
    """
    start = time.time()
    while time.time() - start < timeout:
        try:
            r = requests.get(f"{BASE_URL}/health")
            if r.status_code == 200:
                return
        except requests.exceptions.ConnectionError:
            pass
        time.sleep(1)

    raise RuntimeError("API did not become ready in time")


def test_health():
    wait_for_api()

    r = requests.get(f"{BASE_URL}/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_predict():
    wait_for_api()

    r = requests.post(f"{BASE_URL}/predict", json=VALID_SAMPLE)
    assert r.status_code == 200

    data = r.json()
    assert "prediction" in data
    assert "probability" in data
    assert data["prediction"] in (0, 1)
    assert 0.0 <= data["probability"] <= 1.0


def test_model_info():
    wait_for_api()

    r = requests.get(f"{BASE_URL}/model-info")
    assert r.status_code == 200

    data = r.json()
    assert "model_version" in data
    assert "trained_at" in data
