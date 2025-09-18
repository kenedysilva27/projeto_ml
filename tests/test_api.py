from fastapi.testclient import TestClient
from api import app

client = TestClient(app)

def test_health():
    """Testa se o endpoint /health responde corretamente"""
    response = client.get("/health")
    assert response.status_code == 200
    assert "model_loaded" in response.json()

def test_predict_single():
    """Testa o endpoint /predict com dados irreais"""
    sample = {f"var_{i}": 0.1 for i in range(200)}
    response = client.post("/predict", json=sample)
    assert response.status_code == 200
    data = response.json()
    assert set(data) == {"prediction", "probability", "confidence"}

def test_predict_batch():
    """Testa o endpoint /predict_batch com múltiplos clientes"""
    batch = [{f"var_{i}": 0.1 for i in range(200)} for _ in range(5)]
    response = client.post("/predict_batch", json=batch)
    assert response.status_code == 200
    assert isinstance(response.json(), list)
    assert len(response.json()) == 5
