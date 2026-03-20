import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src.api import app


@pytest.fixture
def sample_input() -> dict:
    path = Path("tests/fixtures/sample_input.json")
    return json.loads(path.read_text(encoding="utf-8"))


def test_health_returns_ok() -> None:
    with TestClient(app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "OK"}


def test_predict_returns_prediction(sample_input: dict) -> None:
    with TestClient(app) as client:
        response = client.post("/predict", json=sample_input)

    assert response.status_code == 200

    body = response.json()
    assert "predicted_price" in body
    assert isinstance(body["predicted_price"], float)
    assert "model_run_id" in body
    assert isinstance(body["model_run_id"], str)


def test_predict_missing_required_field_returns_422(sample_input: dict) -> None:
    bad_input = dict(sample_input)
    bad_input.pop("year")

    with TestClient(app) as client:
        response = client.post("/predict", json=bad_input)

    assert response.status_code == 422
