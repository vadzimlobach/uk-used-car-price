import json
from pathlib import Path

import pytest
from pydantic import ValidationError
from pytest import raises

from src import predict
from src.predict import resolve_latest_model_path, resolve_model_path
from src.schema import CarFeatures


class DummyModel:
    def predict(self, X):
        assert len(X) == 1
        return [12345.67]


class AssertingModel:
    def predict(self, X):
        assert X.shape[0] == 1
        assert "year" in X.columns
        assert "mileage" in X.columns
        return [9999.0]


data = {
    "year": 2018,
    "mileage": 30000,
    "tax": 145,
    "mpg": 55,
    "engineSize": 2.0,
    "brand": "ford",
    "model": "focus",
    "transmission": "manual",
    "fuelType": "petrol",
}

input_data = {
    "year": 2018,
    "mileage": 30000,
    "tax": 145,
    "mpg": 55,
    "engineSize": 2.0,
    "brand": "ford",
    "model": "focus",
    "transmission": "manual",
    "fuelType": "petrol",
}


def test_resolve_latest_model_path_returns_model_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runs_dir = tmp_path / "artifacts" / "runs"
    run_dir = runs_dir / "20260312_081000_rf_baseline"
    run_dir.mkdir(parents=True)

    latest_run_file = runs_dir / "latest_run.txt"
    latest_run_file.write_text("20260312_081000_rf_baseline", encoding="utf-8")

    model_file = run_dir / "model.joblib"
    model_file.write_text("dummy-model", encoding="utf-8")

    monkeypatch.chdir(tmp_path)

    resolved = resolve_latest_model_path()

    assert resolved.resolve() == model_file.resolve()


def test_resolve_latest_model_path_raises_if_latest_run_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)

    with pytest.raises(SystemExit, match="latest_run.txt not found"):
        resolve_latest_model_path()


def test_resolve_latest_model_path_raises_if_model_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runs_dir = tmp_path / "artifacts" / "runs"
    runs_dir.mkdir(parents=True)

    latest_run_file = runs_dir / "latest_run.txt"
    latest_run_file.write_text("20260312_081000_rf_baseline", encoding="utf-8")

    monkeypatch.chdir(tmp_path)

    with pytest.raises(SystemExit, match="Model not found"):
        resolve_latest_model_path()


def test_resolve_model_path_prefers_cli_argument(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cli_model = tmp_path / "cli_model.joblib"
    cli_model.write_text("dummy-model", encoding="utf-8")

    env_model = tmp_path / "env_model.joblib"
    env_model.write_text("dummy-model", encoding="utf-8")

    monkeypatch.setenv("MODEL_PATH", str(env_model))

    resolved = resolve_model_path(cli_model)

    assert resolved == cli_model


def test_resolve_model_path_uses_env_when_cli_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env_model = tmp_path / "env_model.joblib"
    env_model.write_text("dummy-model", encoding="utf-8")

    monkeypatch.setenv("MODEL_PATH", str(env_model))

    resolved = resolve_model_path(None)

    assert resolved == env_model


def test_resolve_model_path_falls_back_to_latest_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runs_dir = tmp_path / "artifacts" / "runs"
    run_dir = runs_dir / "20260312_081000_rf_baseline"
    run_dir.mkdir(parents=True)

    latest_run_file = runs_dir / "latest_run.txt"
    latest_run_file.write_text("20260312_081000_rf_baseline", encoding="utf-8")

    model_file = run_dir / "model.joblib"
    model_file.write_text("dummy-model", encoding="utf-8")

    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("MODEL_PATH", raising=False)

    resolved = resolve_model_path(None)

    assert resolved.resolve() == model_file.resolve()


def test_resolve_model_path_raises_if_env_model_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    missing_model = tmp_path / "missing_model.joblib"
    monkeypatch.setenv("MODEL_PATH", str(missing_model))

    with pytest.raises(SystemExit, match="MODEL_PATH is set but file not found"):
        resolve_model_path(None)


def test_resolve_model_path_raises_if_cli_model_missing(tmp_path: Path) -> None:
    missing_model = tmp_path / "missing_model.joblib"

    with pytest.raises(SystemExit, match="Model not found"):
        resolve_model_path(missing_model)


def test_schema_valid_input():
    car_data = CarFeatures(**data)
    assert car_data.year == 2018
    assert car_data.mileage == 30000
    assert car_data.transmission == "manual"


def test_schema_missing_input_field():
    data = {
        "year": 2018,
        "mileage": 30000,
        "tax": 145,
        "mpg": 45,
        "engineSize": 2.0,
        "brand": "ford",
        "model": "focus",
        "transmission": "manual",
    }

    with raises(ValidationError):
        CarFeatures(**data)


def test_schema_wrong_type():
    data = {
        "year": 2018,
        "mileage": "thirty thousand",
        "tax": 145,
        "mpg": 45,
        "engineSize": 2.0,
        "brand": "ford",
        "model": "focus",
        "transmission": "manual",
    }
    with raises(ValidationError):
        CarFeatures(**data)


def test_schema_from_json_fixture():
    with open("tests/fixtures/sample_input.json") as f:
        data = json.load(f)

    car_obj = CarFeatures(**data)
    assert car_obj.brand.lower() == "ford"
    assert car_obj.model.lower() == "focus"
    assert car_obj.transmission.lower() == "manual"
    assert car_obj.fuelType.lower() == "petrol"
    assert car_obj.year == 2018
    assert car_obj.mileage == 35000
    assert car_obj.tax == 150
    assert car_obj.mpg == 45.0
    assert car_obj.engineSize == 1.0


@pytest.mark.parametrize(
    "field,value",
    [
        ("year", 1949),
        ("year", 2027),
        ("mileage", -1),
        ("tax", -1),
        ("mpg", -1),
        ("engineSize", -0.1),
    ],
)
def test_schema_numeric_boundaries(field, value):

    data[field] = value

    with raises(ValidationError):
        CarFeatures(**data)


def test_schema_rejects_extra_fields():
    data = {
        "year": 2018,
        "mileage": 30000,
        "tax": 145,
        "mpg": 55,
        "engineSize": 2.0,
        "brand": "ford",
        "model": "focus",
        "transmission": "manual",
        "fuelType": "petrol",
        "colour": "blue",
    }

    with pytest.raises(ValidationError):
        CarFeatures(**data)


def test_predict_main_happy_path(tmp_path, monkeypatch, capsys):
    model_path = tmp_path / "model.joblib"
    model_path.write_text("dummy-model", encoding="utf-8")
    input_path = tmp_path / "sample_input.json"
    input_path.write_text(json.dumps(input_data), encoding="utf-8")

    monkeypatch.setattr(predict.joblib, "load", lambda _: DummyModel())
    monkeypatch.setattr(
        predict,
        "add_features",
        lambda *args, **kwargs: args[0],
    )

    monkeypatch.setattr(
        "sys.argv",
        [
            "predict",
            "--model",
            str(model_path),
            "--input",
            str(input_path),
            "--config",
            "configs/train.yaml",
        ],
    )

    predict.main()

    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert "predicted_price" in payload
    assert isinstance(payload["predicted_price"], float)
    assert payload["predicted_price"] == 12345.67


def test_predict_main_wrong_type(tmp_path, monkeypatch):
    input_data = {
        "year": 2018,
        "mileage": "thirty thousand",
        "tax": 145,
        "mpg": 55.0,
        "engineSize": 2.0,
        "brand": "ford",
        "model": "focus",
        "transmission": "manual",
        "fuelType": "petrol",
    }

    input_path = tmp_path / "bad_type_input.json"
    input_path.write_text(json.dumps(input_data), encoding="utf-8")

    monkeypatch.setattr(predict.joblib, "load", lambda _: DummyModel())
    monkeypatch.setattr(
        "sys.argv",
        [
            "predict",
            "--model",
            "fake.joblib",
            "--input",
            str(input_path),
            "--config",
            "configs/train.yaml",
        ],
    )

    with pytest.raises(SystemExit):
        predict.main()


def test_predict_passes_single_row_dataframe(tmp_path, monkeypatch, capsys):
    model_path = tmp_path / "fake.joblib"
    model_path.write_text("dummy-model", encoding="utf-8")

    input_path = tmp_path / "sample_input.json"
    input_path.write_text(json.dumps(input_data), encoding="utf-8")

    monkeypatch.setattr(predict.joblib, "load", lambda _: AssertingModel())
    monkeypatch.setattr(
        predict,
        "add_features",
        lambda *args, **kwargs: args[0],
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "predict",
            "--model",
            str(model_path),
            "--input",
            str(input_path),
            "--config",
            "configs/train.yaml",
        ],
    )

    predict.main()

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert payload["predicted_price"] == 9999.0
