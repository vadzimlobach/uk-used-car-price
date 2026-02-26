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


# -----------------------
# clean_price() tests
# -----------------------

def test_clean_price_converts_currency_strings_to_numbers():
    s = pd.Series(["£1,234", "  999 ", "£0", "bad", None])
    out = clean_price(s)
    # "bad" and None -> NaN
    assert out.tolist()[0] == 1234.0
    assert out.tolist()[1] == 999
    assert out.tolist()[2] == 0
    assert pd.isna(out.tolist()[3])
    assert pd.isna(out.tolist()[4])


def test_clean_price_keeps_numeric_series_numeric():
    s = pd.Series([100, 200, None])
    out = clean_price(s)
    assert out.dtype.kind in ("i", "f")  # int/float
    assert out.tolist()[0] == 100
    assert out.tolist()[1] == 200
    assert pd.isna(out.tolist()[2])


# -----------------------
# preprocess_data() tests
# -----------------------

def test_preprocess_data_raises_when_target_missing(logger):
    df = pd.DataFrame({"model": ["a"], "year": [2020]})
    with pytest.raises(ValueError):
        preprocess_data(df, target="price", logger=logger)


def test_preprocess_drops_all_nan_columns(logger):
    df = pd.DataFrame({
        "price": ["£1000", "£2000"],
        "all_nan": [None, None],
        "model": ["a", "b"],
    })
    out = preprocess_data(df, target="price", logger=logger)
    assert "all_nan" not in out.columns


def test_preprocess_drops_duplicate_rows(logger):
    df = pd.DataFrame({
        "price": ["£1000", "£1000"],
        "model": ["a", "a"],
        "year": [2020, 2020],
    })
    out = preprocess_data(df, target="price", logger=logger)
    assert out.shape[0] == 1


def test_preprocess_removes_invalid_target_rows(logger):
    df = pd.DataFrame({
        "price": ["£1000", "bad", "-50", None],
        "model": ["a", "b", "c", "d"],
    })
    out = preprocess_data(df, target="price", logger=logger)

    # Should keep only £1000, drop bad/None, and drop negative
    assert out.shape[0] == 1
    assert out["price"].iloc[0] == 1000


def test_preprocess_fills_numeric_missing_with_median(logger):
    df = pd.DataFrame({
        "price": ["£1000", "£2000", "£3000"],
        "mileage": [10_000, None, 30_000],  # median is 20_000
        "year": [2020, 2019, 2018],
    })
    out = preprocess_data(df, target="price", logger=logger)

    assert out["mileage"].isna().sum() == 0
    assert out["mileage"].tolist() == [10_000, 20_000, 30_000]


def test_preprocess_does_not_fill_target_with_median(logger):
    # target NaNs should be dropped, not filled
    df = pd.DataFrame({
        "price": ["£1000", None, "£3000"],
        "mileage": [10_000, 20_000, 30_000],
    })
    out = preprocess_data(df, target="price", logger=logger)

    assert out.shape[0] == 2
    assert out["price"].tolist() == [1000, 3000]


def test_preprocess_fills_categorical_missing_with_unknown(logger):
    df = pd.DataFrame({
        "price": ["£1000", "£2000"],
        "transmission": ["Manual", None],
        "fuelType": [None, "Petrol"],
    })
    out = preprocess_data(df, target="price", logger=logger)

    assert out["transmission"].tolist() == ["Manual", "Unknown"]
    assert out["fuelType"].tolist() == ["Unknown", "Petrol"]
