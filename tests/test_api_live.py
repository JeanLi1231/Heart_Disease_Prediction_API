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
    "ST_Slope": "Up",
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


def test_predict_returns_503_when_model_missing():
    """
    In CI we do not ship model.pkl into the container.
    The service should still start, but /predict should return 503.
    """
    wait_for_api()

    r = requests.post(f"{BASE_URL}/v1/predict", json=VALID_SAMPLE)
    assert r.status_code == 503

    data = r.json()
    assert "detail" in data


def test_model_info_when_model_missing():
    """
    When no model is loaded, /model-info should still respond (usually {}).
    """
    wait_for_api()

    r = requests.get(f"{BASE_URL}/v1/model-info")
    assert r.status_code == 200

    data = r.json()
    assert isinstance(data, dict)
