from __future__ import annotations

from pathlib import Path
import pandas as pd
import pytest


# Update this import to match your file/module name
from src.data_loader import (
    load_data,
    combine_csv_files,
    standardize_columns,
    coalesce_columns,
    basic_report,
    TARGET_COLS,
)


class DummyLogger:
    """Minimal logger stub so we don't depend on logging configuration in unit tests."""
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


def write_csv(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


# -----------------------
# load_data() tests
# -----------------------

def test_load_data_file_not_found(tmp_path: Path, logger):
    missing = tmp_path / "nope.csv"
    with pytest.raises(FileNotFoundError):
        load_data(missing, logger)


def test_load_data_not_a_file(tmp_path: Path, logger):
    # path exists but is a directory
    with pytest.raises(ValueError):
        load_data(tmp_path, logger)


def test_load_data_not_csv_suffix(tmp_path: Path, logger):
    p = tmp_path / "file.txt"
    p.write_text("hello")
    with pytest.raises(ValueError):
        load_data(p, logger)


def test_load_data_success(tmp_path: Path, logger):
    p = tmp_path / "cars.csv"
    df = pd.DataFrame({"model": ["a"], "price": [1000]})
    write_csv(p, df)

    out = load_data(p, logger)
    assert isinstance(out, pd.DataFrame)
    assert out.shape == (1, 2)
    assert list(out.columns) == ["model", "price"]


# -----------------------
# coalesce_columns() tests
# -----------------------

def test_coalesce_columns_secondary_missing_no_change(logger):
    df = pd.DataFrame({"mileage": [1, None, 3]})
    out = coalesce_columns(df.copy(), "mileage", "mileage2", logger)
    pd.testing.assert_frame_equal(out, df)


def test_coalesce_columns_fills_missing(logger):
    df = pd.DataFrame({"mileage": [1, None, 3], "mileage2": [10, 20, 30]})
    out = coalesce_columns(df, "mileage", "mileage2", logger)
    assert out["mileage"].tolist() == [1, 20, 3]


def test_coalesce_columns_primary_missing_created(logger):
    df = pd.DataFrame({"mileage2": [10, 20]})
    out = coalesce_columns(df, "mileage", "mileage2", logger)
    assert "mileage" in out.columns
    assert out["mileage"].tolist() == [10, 20]


# -----------------------
# standardize_columns() tests
# -----------------------

def test_standardize_columns_renames_variants_and_keeps_target_schema(logger):
    df = pd.DataFrame({
        "model": ["a"],
        "year": [2019],
        "price": [1000],
        "transmission": ["Manual"],
        "mileage": [50000],
        "fuel type": ["Petrol"],         # variant
        "tax(£)": [150],                 # variant
        "mpg": [55.0],
        "engine size": [1.6],            # variant
        "reference": ["XYZ"],            # extra should be dropped
    })

    out = standardize_columns(df, logger)

    assert list(out.columns) == TARGET_COLS
    assert out.loc[0, "fuelType"] == "Petrol"
    assert out.loc[0, "engineSize"] == 1.6
    assert out.loc[0, "tax"] == 150


def test_standardize_columns_coalesces_secondary_fields(logger):
    df = pd.DataFrame({
        "model": ["a"],
        "year": [2019],
        "price": [1000],
        "transmission": ["Manual"],
        "mileage": [None],
        "mileage2": [12345],
        "fuelType": [None],
        "fuelType2": ["Diesel"],
        "engineSize": [None],
        "engineSize2": [2.0],
    })

    out = standardize_columns(df, logger)
    assert out.loc[0, "mileage"] == 12345
    assert out.loc[0, "fuelType"] == "Diesel"
    assert out.loc[0, "engineSize"] == 2.0


def test_standardize_columns_adds_missing_columns_as_na(logger):
    # minimal columns
    df = pd.DataFrame({"model": ["a"], "year": [2019], "price": [1000]})
    out = standardize_columns(df, logger)

    assert list(out.columns) == TARGET_COLS
    # columns not provided should exist and be NA
    assert pd.isna(out.loc[0, "tax"])
    assert pd.isna(out.loc[0, "mpg"])
    assert pd.isna(out.loc[0, "engineSize"])


# -----------------------
# combine_csv_files() tests
# -----------------------

def test_combine_csv_files_directory_missing(tmp_path: Path, logger):
    with pytest.raises(FileNotFoundError):
        combine_csv_files(tmp_path / "missing_dir", logger)


def test_combine_csv_files_not_a_directory(tmp_path: Path, logger):
    p = tmp_path / "file.csv"
    p.write_text("x")
    with pytest.raises(ValueError):
        combine_csv_files(p, logger)


def test_combine_csv_files_no_csv_files(tmp_path: Path, logger):
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    with pytest.raises(ValueError):
        combine_csv_files(empty_dir, logger)


def test_combine_csv_files_success_and_standardizes(tmp_path: Path, logger):
    d = tmp_path / "data"
    d.mkdir()

    df1 = pd.DataFrame({
        "model": ["a"],
        "year": [2019],
        "price": [1000],
        "transmission": ["Manual"],
        "mileage": [50000],
        "fuelType": ["Petrol"],
        "tax": [150],
        "mpg": [55.0],
        "engineSize": [1.6],
    })

    df2 = pd.DataFrame({
        "model": ["b"],
        "year": [2018],
        "price": [900],
        "transmission": ["Auto"],
        "mileage": [None],
        "mileage2": [40000],
        "fuel type": ["Diesel"],
        "engine size": [2.0],
        "reference": ["X"],
    })

    write_csv(d / "a.csv", df1)
    write_csv(d / "b.csv", df2)

    out = combine_csv_files(d, logger)

    assert list(out.columns) == TARGET_COLS
    assert out.shape[0] == 2
    assert out.loc[out["model"] == "b", "mileage"].iloc[0] == 40000
    
def test_combine_csv_files_exclude_combined_csv(tmp_path: Path, logger):
    d = tmp_path / "data"
    d.mkdir()

    df1 = pd.DataFrame({
        "model": ["a4", "a3"],
        "year": [2019, 2018],
        "price": [1000, 5000],
        "transmission": ["Manual", "Manual"],
        "mileage": [50000, 35000],
        "fuelType": ["Petrol", "Diesel"],
        "tax": [150, 200],
        "mpg": [55.0, 45.0],
        "engineSize": [1.6, 2.0],
    })

    df2 = pd.DataFrame({
        "model": ["3 series", "3 series", "X5"],
        "year": [2018, 2018, 2019],
        "price": [900, 5700, 15000],
        "transmission": ["Auto", "Auto", "Auto"],
        "mileage2": [40000, 45000, 55000],
        "fuel type": ["Diesel", "Diesel", "Petrol"],
        "engine size": [2.0, 2.0, 1.8],
    })

    df3 = pd.DataFrame({
        "model": ["Golf", "Tiguan", "Touareg"],
        "year": [2019, 2018, 2017],
        "price": [1000, 5000, 8000],
        "transmission": ["Manual", "Manual", "Manual"],
        "mileage": [50000, 35000, 45000],
        "fuelType": ["Petrol", "Diesel", "Diesel"],
        "tax": [150, 200, 250],
        "mpg": [55.0, 45.0, 40.0],
        "engineSize": [1.6, 2.0, 1.8],
    })

    write_csv(d / "a.csv", df1)
    write_csv(d / "b.csv", df2)
    write_csv(d / "combined.csv", df3)

    out = combine_csv_files(d, logger)

    assert list(out.columns) == TARGET_COLS
    assert out.shape[0] == 5  # 2 from df1, 3 from df2 (mileage filled from mileage2), none from combined.csv
    assert out.loc[out["model"] == "3 series", "mileage"].tolist() == [40000, 45000]
    assert out.loc[out["model"] == "a4", "mileage"].tolist() == [50000]


# -----------------------
# basic_report() tests
# -----------------------

def test_basic_report_raises_if_target_missing(logger):
    df = pd.DataFrame({"x": [1, 2]})
    with pytest.raises(ValueError):
        basic_report(df, target="price", logger=logger)


def test_basic_report_runs_on_valid_df(logger):
    df = pd.DataFrame({"price": [1000, 2000], "model": ["a", "b"]})
    # should not raise
    basic_report(df, target="price", logger=logger)
