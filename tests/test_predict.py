import json
import pytest
from pydantic import ValidationError
from pytest import raises
from pathlib import Path
from src.schema import CarFeatures
from src import predict

class DummyModel():
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

def test_schema_valid_input():
    car_data = CarFeatures(**data)
    assert car_data.year == 2018
    assert car_data.mileage == 30000
    assert car_data.transmission == 'manual'

def test_schema_missing_input_field():
    data = {
        'year': 2018,
        "mileage": 30000,
        "tax": 145,
        "mpg": 45,
        "engineSize": 2.0,
        "brand": "ford",
        "model": "focus",
        "transmission": "manual"
    }

    with raises(ValidationError):
        CarFeatures(**data)

def test_schema_wrong_type():
    data = {
        'year': 2018,
        "mileage": 'thirty thousand',
        "tax": 145,
        "mpg": 45,
        "engineSize": 2.0,
        "brand": "ford",
        "model": "focus",
        "transmission": "manual"
    }
    with raises(ValidationError):
        CarFeatures(**data)

def test_schema_from_json_fixture():
    with open('tests/fixtures/sample_input.json') as f:
        data = json.load(f)

    car_obj = CarFeatures(**data)
    assert car_obj.brand.lower() == 'ford'
    assert car_obj.model.lower() == 'focus'
    assert car_obj.transmission.lower() == 'manual'
    assert car_obj.fuelType.lower() == 'petrol'
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
            "artifacts/runs/fake/model.joblib",
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
            "fake.joblib",
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
