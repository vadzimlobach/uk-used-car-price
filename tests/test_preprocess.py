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
    # Default config for most tests
    return {
        "allow_zero_price": False,
        "categorical_fill_value": "Unknown",
        "numeric_fill_strategy": "median",
        "invalid_target_threshold": 0.1,
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

    assert out.dtype.kind in ("i", "f")  # int/float
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
    })
    out = preprocess_data(df, target="price", logger=logger, config=test_config)
    assert "all_nan" not in out.columns


def test_preprocess_drops_duplicate_rows(logger, test_config):
    df = pd.DataFrame({
        "price": ["£1000", "£1000"],
        "model": ["a", "a"],
        "year": [2020, 2020],
    })
    out = preprocess_data(df, target="price", logger=logger, config=test_config)
    assert out.shape[0] == 1


def test_preprocess_removes_invalid_target_rows_disallow_zero(logger, test_config):
    # allow_zero_price is False in test_config
    df = pd.DataFrame({
        "price": ["£1000", "bad", "-50", None, "£0"],
        "model": ["a", "b", "c", "d", "e"],
    })
    out = preprocess_data(df, target="price", logger=logger, config=test_config)

    # Should keep only £1000, drop bad/None, drop negative, and drop zero
    assert out.shape[0] == 1
    assert out["price"].iloc[0] == 1000


def test_preprocess_keeps_zero_target_when_allowed(logger, test_config):
    cfg = dict(test_config)
    cfg["allow_zero_price"] = True

    df = pd.DataFrame({
        "price": ["£1000", "£0", "-50", None],
        "model": ["a", "b", "c", "d"],
    })
    out = preprocess_data(df, target="price", logger=logger, config=cfg)

    # Should keep £1000 and £0, drop negative and NaN
    assert out.shape[0] == 2
    assert sorted(out["price"].tolist()) == [0, 1000]


def test_preprocess_fills_numeric_missing_with_median(logger, test_config):
    df = pd.DataFrame({
        "price": ["£1000", "£2000", "£3000"],
        "mileage": [10_000, None, 30_000],  # median is 20_000
        "year": [2020, 2019, 2018],
    })
    out = preprocess_data(df, target="price", logger=logger, config=test_config)

    assert out["mileage"].isna().sum() == 0
    assert out["mileage"].tolist() == [10_000, 20_000, 30_000]


def test_preprocess_fills_numeric_missing_with_mean(logger, test_config):
    cfg = dict(test_config)
    cfg["numeric_fill_strategy"] = "mean"

    df = pd.DataFrame({
        "price": ["£1000", "£2000", "£3000"],
        "mileage": [10_000, None, 30_000],  # mean is 20_000
    })
    out = preprocess_data(df, target="price", logger=logger, config=cfg)

    assert out["mileage"].isna().sum() == 0
    assert out["mileage"].tolist() == [10_000, 20_000, 30_000]


def test_preprocess_does_not_fill_target_with_imputer(logger, test_config):
    # target NaNs should be dropped, not filled
    df = pd.DataFrame({
        "price": ["£1000", None, "£3000"],
        "mileage": [10_000, 20_000, 30_000],
    })
    out = preprocess_data(df, target="price", logger=logger, config=test_config)

    assert out.shape[0] == 2
    assert out["price"].tolist() == [1000, 3000]


def test_preprocess_fills_categorical_missing_with_unknown(logger, test_config):
    df = pd.DataFrame({
        "price": ["£1000", "£2000"],
        "transmission": ["Manual", None],
        "fuelType": [None, "Petrol"],
    })
    out = preprocess_data(df, target="price", logger=logger, config=test_config)

    assert out["transmission"].tolist() == ["Manual", "Unknown"]
    assert out["fuelType"].tolist() == ["Unknown", "Petrol"]


def test_preprocess_uses_custom_categorical_fill_value(logger, test_config):
    cfg = dict(test_config)
    cfg["categorical_fill_value"] = "MISSING"

    df = pd.DataFrame({
        "price": ["£1000", "£2000"],
        "transmission": ["Manual", None],
    })
    out = preprocess_data(df, target="price", logger=logger, config=cfg)

    assert out["transmission"].tolist() == ["Manual", "MISSING"]