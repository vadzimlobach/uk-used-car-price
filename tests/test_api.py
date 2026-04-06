import json
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src.api import app


class DummyModel:
    def predict(self, X) -> list[float]:
        return [12345.0]


@pytest.fixture
def sample_input() -> dict:
    path = Path("tests/fixtures/sample_input.json")
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture
def api_client(monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    monkeypatch.setattr("src.api.resolve_latest_model_path", lambda: Path("dummy/model.joblib"))
    monkeypatch.setattr("src.api.joblib.load", lambda _: DummyModel())
    # monkeypatch.setattr("src.api.add_features", lambda X, logger, config: X)

    with TestClient(app) as test_client:
        yield test_client


def test_health_returns_ok(api_client: TestClient) -> None:
    response = api_client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "OK"}


def test_predict_returns_prediction(api_client: TestClient, sample_input: dict) -> None:
    response = api_client.post("/predict", json=sample_input)

    assert response.status_code == 200

    body = response.json()
    model_version = body["model_version"]
    assert "predicted_price" in body
    assert isinstance(body["predicted_price"], float)
    assert "run_id" in model_version
    assert isinstance(model_version["run_id"], str)


def test_predict_missing_required_field_returns_422(
    api_client: TestClient, sample_input: dict
) -> None:
    bad_input = dict(sample_input)
    bad_input.pop("year")

    response = api_client.post("/predict", json=bad_input)

    assert response.status_code == 422
