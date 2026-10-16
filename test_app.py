from fastapi.testclient import TestClient
from app import app

client = TestClient(app)

def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True

def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    assert "/predict" in response.json()["predict"]

def test_predict_schema_and_probabilities():
    response = client.post("/predict", json={"texts": ["I absolutely love this", "This is awful"]})
    assert response.status_code == 200
    body = response.json()
    assert body["model_version"] == "1.0.0"
    assert len(body["predictions"]) == 2
    for item in body["predictions"]:
        assert item["label"] in {"Positive", "Negative"}
        assert 0.0 <= item["confidence"] <= 1.0
        assert set(item["probabilities"]) == {"Positive", "Negative"}
        assert abs(sum(item["probabilities"].values()) - 1.0) < 1e-4

def test_empty_text_rejected():
    response = client.post("/predict", json={"texts": ["   "]})
    assert response.status_code == 422

def test_missing_texts_rejected():
    response = client.post("/predict", json={})
    assert response.status_code == 422
