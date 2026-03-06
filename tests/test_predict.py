import json
import pytest
from pydantic import ValidationError
from pytest import raises
from pathlib import Path
from src.schema import CarFeatures

def test_schema_valid_input():
    data = {
        'year': 2018,
        "mileage": 30000,
        "tax": 145,
        "mpg": 45,
        "engineSize": 2.0,
        "brand": "ford",
        "model": "focus",
        "transmission": "manual",
        "fuelType": "petrol"
    }

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