from pathlib import Path
import pandas as pd
import pytest
from src.preprocess import read_data_from_file, save_preprocessed_data

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
# -----------------------
# read_data_from_file tests
# -----------------------

def test_read_data_from_file_success(tmp_path: Path, logger):
    # Arrange
    input_path = tmp_path / "input.csv"
    df = pd.DataFrame({
        "price": [1000, 2000],
        "model": ["a", "b"],
    })
    df.to_csv(input_path, index=False)

    # Act
    result = read_data_from_file(input_path, logger)

    # Assert
    assert isinstance(result, pd.DataFrame)
    pd.testing.assert_frame_equal(result, df)


# -----------------------
# save_preprocessed_data tests
# -----------------------

def test_save_preprocessed_data_creates_file(tmp_path: Path, logger):
    # Arrange
    df = pd.DataFrame({
        "price": [1000, 2000],
        "model": ["a", "b"],
    })
    output_path = tmp_path / "nested" / "folder" / "output.csv"

    # Act
    save_preprocessed_data(df, output_path, logger)

    # Assert
    assert output_path.exists()

    saved_df = pd.read_csv(output_path)
    pd.testing.assert_frame_equal(saved_df, df)