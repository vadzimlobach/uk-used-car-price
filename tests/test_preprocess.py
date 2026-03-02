from __future__ import annotations

from pathlib import Path
import pandas as pd
import pytest

from src.preprocess import clean_price, preprocess_data


class DummyLogger:
    def info(self, *args, **kwargs):
        pass
    def warning(self, *args, **kwargs):
        pass
    def error(self, *args, **kwargs):
        pass
    def debug(self, *args, **kwargs):
        pass
    def exception(self, *args, **kwargs):
        pass


@pytest.fixture()
def logger():
    return DummyLogger()


@pytest.fixture()
def test_config():
    # Keep deterministic reference year for feature tests
    return {
        "allow_zero_price": False,
        "categorical_fill_value": "Unknown",
        "numeric_fill_strategy": "median",
        "invalid_target_threshold": 0.1,
        "reference_year": 2026,
        "min_year": 1970,
        "max_year": 2026,
        "max_mileage_per_year": 50000,
    }


# -----------------------
# clean_price() tests
# -----------------------

def test_clean_price_converts_currency_strings_to_numbers():
    s = pd.Series(["£1,234", "  999 ", "£0", "bad", None])
    out = clean_price(s)

    assert out.iloc[0] == 1234
    assert out.iloc[1] == 999
    assert out.iloc[2] == 0
    assert pd.isna(out.iloc[3])
    assert pd.isna(out.iloc[4])


def test_clean_price_keeps_numeric_series_numeric():
    s = pd.Series([100, 200, None])
    out = clean_price(s)

    assert out.dtype.kind in ("i", "f")
    assert out.iloc[0] == 100
    assert out.iloc[1] == 200
    assert pd.isna(out.iloc[2])


# -----------------------
# preprocess_data() tests
# -----------------------

def test_preprocess_data_raises_when_target_missing(logger, test_config):
    df = pd.DataFrame({"model": ["a"], "year": [2020]})
    with pytest.raises(ValueError):
        preprocess_data(df, target="price", logger=logger, config=test_config)


def test_preprocess_drops_all_nan_columns(logger, test_config):
    df = pd.DataFrame({
        "price": ["£1000", "£2000"],
        "all_nan": [None, None],
        "model": ["a", "b"],
        "year": [2020, 2020],
        "mileage": [10000, 20000],
    })
    out = preprocess_data(df, target="price", logger=logger, config=test_config)
    assert "all_nan" not in out.columns


def test_preprocess_drops_duplicate_rows(logger, test_config):
    df = pd.DataFrame({
        "price": ["£1000", "£1000"],
        "model": ["a", "a"],
        "year": [2020, 2020],
        "mileage": [10000, 10000],
    })
    out = preprocess_data(df, target="price", logger=logger, config=test_config)
    assert out.shape[0] == 1


def test_preprocess_removes_invalid_target_rows_disallow_zero(logger, test_config):
    df = pd.DataFrame({
        "price": ["£1000", "bad", "-50", None, "£0"],
        "model": ["a", "b", "c", "d", "e"],
        "year": [2020, 2020, 2020, 2020, 2020],
        "mileage": [10000, 10000, 10000, 10000, 10000],
    })
    out = preprocess_data(df, target="price", logger=logger, config=test_config)

    assert out.shape[0] == 1
    assert out["price"].iloc[0] == 1000


def test_preprocess_keeps_zero_price_when_allowed(logger, test_config):
    cfg = dict(test_config)
    cfg["allow_zero_price"] = True

    df = pd.DataFrame({
        "price": ["£1000", "£0", "-50", None],
        "model": ["a", "b", "c", "d"],
        "year": [2020, 2020, 2020, 2020],
        "mileage": [10000, 10000, 10000, 10000],
    })
    out = preprocess_data(df, target="price", logger=logger, config=cfg)

    assert out.shape[0] == 2
    assert sorted(out["price"].tolist()) == [0, 1000]


def test_preprocess_fills_numeric_missing_with_median(logger, test_config):
    df = pd.DataFrame({
        "price": ["£1000", "£2000", "£3000"],
        "mileage": [10_000, None, 30_000],
        "year": [2020, 2019, 2018],
        "tax": [10, None, 30],  # numeric imputation should fill
    })
    out = preprocess_data(df, target="price", logger=logger, config=test_config)

    assert out["tax"].isna().sum() == 0


def test_preprocess_fills_categorical_missing_with_unknown(logger, test_config):
    df = pd.DataFrame({
        "price": ["£1000", "£2000"],
        "transmission": ["Manual", None],
        "fuelType": [None, "Petrol"],
        "year": [2020, 2020],
        "mileage": [10000, 20000],
    })
    out = preprocess_data(df, target="price", logger=logger, config=test_config)

    assert out["transmission"].tolist() == ["Manual", "Unknown"]
    assert out["fuelType"].tolist() == ["Unknown", "Petrol"]


# -----------------------
# Feature engineering tests (add_features via preprocess_data)
# -----------------------

def test_preprocess_adds_feature_columns(logger, test_config):
    df = pd.DataFrame({
        "price": ["£1000", "£2000"],
        "year": [2020, 2019],
        "mileage": [12000, 24000],
    })
    out = preprocess_data(df, target="price", logger=logger, config=test_config)

    assert "car_age" in out.columns
    assert "mileage_per_year" in out.columns


def test_car_age_is_correct(logger, test_config):
    df = pd.DataFrame({
        "price": ["£1000"],
        "year": [2020],
        "mileage": [12000],
    })
    out = preprocess_data(df, target="price", logger=logger, config=test_config)

    assert out["car_age"].iloc[0] == 6  # 2026 - 2020


def test_future_year_is_dropped(logger, test_config):
    df = pd.DataFrame({
        "price": ["£1000", "£2000"],
        "year": [2020, 2060],  # 2060 is outside max_year
        "mileage": [12000, 12000],
    })
    out = preprocess_data(df, target="price", logger=logger, config=test_config)

    assert out.shape[0] == 1
    assert "year" not in out.columns  # year is intentionally dropped after deriving car_age


def test_mileage_string_is_cleaned_and_feature_runs(logger, test_config):
    df = pd.DataFrame({
        "price": ["£1000", "£2000"],
        "year": [2020, 2020],
        "mileage": ["12,000", " 24000 "],  # strings
    })
    out = preprocess_data(df, target="price", logger=logger, config=test_config)

    # Should not crash, should compute features
    assert out.shape[0] == 2
    assert out["mileage_per_year"].notna().all()


def test_zero_mileage_is_dropped(logger, test_config):
    df = pd.DataFrame({
        "price": ["£1000", "£2000"],
        "year": [2020, 2020],
        "mileage": [0, 12000],  # 0 should be dropped (drop_zero=True)
    })
    out = preprocess_data(df, target="price", logger=logger, config=test_config)

    assert out.shape[0] == 1
    assert out["mileage"].iloc[0] == 12000


def test_mileage_per_year_outlier_is_dropped(logger, test_config):
    cfg = dict(test_config)
    cfg["max_mileage_per_year"] = 50000

    df = pd.DataFrame({
        "price": ["£1000", "£2000"],
        "year": [2025, 2025],           # age=1
        "mileage": [12000, 200000],     # mileage_per_year = 12k vs 200k
    })
    out = preprocess_data(df, target="price", logger=logger, config=cfg)

    assert out.shape[0] == 1
    assert out["mileage"].iloc[0] == 12000
