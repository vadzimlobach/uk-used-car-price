import argparse
from pathlib import Path

import pandas as pd
from pandas.api.types import is_object_dtype, is_string_dtype

from src.config import load_config
from src.data_io import read_data_from_file, save_preprocessed_data
from src.logging_config import setup_logging


def preprocess_data(df: pd.DataFrame, target: str, logger, config: dict) -> pd.DataFrame:
    """Preprocess the data by cleaning the target column."""

    validate_column_exists(df, target, logger)

    logger.info("Starting preprocess. Raw shape=%s", df.shape)

    df = drop_empty_columns(df, logger)
    df = remove_duplicate_rows(df, logger)
    # Clean the target column
    df[target] = clean_price(df[target])
    df = remove_invalid_target_rows(df, target, logger, config)
    df["mileage"] = clean_numeric_column(df["mileage"])
    df = add_features(df, logger, config)

    logger.info("Finished preprocess. Cleaned shape=%s", df.shape)

    return df


def add_features(df: pd.DataFrame, logger, config: dict) -> pd.DataFrame:
    logger.info("Adding derived features.")
    validate_column_exists(df, "year", logger)
    validate_column_exists(df, "mileage", logger)
    config = config["data"]

    rows_start = df.shape[0]

    df = drop_rows_with_missing_target(df, "year", logger)
    df = drop_rows_not_in_range(df, "year", config["min_year"], config["max_year"], logger)
    car_age = config["reference_year"] - df["year"]
    df = df.copy()  # Avoid SettingWithCopyWarning
    df["car_age"] = car_age
    df = df.drop(columns=["year"])  # Drop original year column after creating car_age
    logger.info("%s rows dropped due to car age rules.", rows_start - df.shape[0])
    logger.info("min car age: %s, max car age: %s", df["car_age"].min(), df["car_age"].max())

    before_rows = df.shape[0]
    df = drop_rows_with_missing_target(df, "mileage", logger)
    df = drop_rows_with_negative_target(df, "mileage", logger, drop_zero=True)
    age_for_division = df["car_age"].clip(lower=1)  # Avoid division by zero for new cars
    df = df.copy()  # Avoid SettingWithCopyWarning
    df["mileage_per_year"] = df["mileage"] / age_for_division
    df = drop_rows_not_in_range(df, "mileage_per_year", 0, config["max_mileage_per_year"], logger)
    logger.info("Dropped %s rows due to mileage rules", before_rows - df.shape[0])
    logger.info(
        "min mileage_per_year: %s, max mileage_per_year: %s",
        df["mileage_per_year"].min(),
        df["mileage_per_year"].max(),
    )

    rows_end = df.shape[0]

    logger.info(
        "Added derived features. Rows before: %s, after: %s, dropped: %s",
        rows_start,
        rows_end,
        rows_start - rows_end,
    )
    return df


def clean_price(series: pd.Series) -> pd.Series:
    """Clean the price column by removing currency symbols and commas."""
    if is_string_dtype(series):
        series = (
            series.astype(str)
            .str.replace("£", "", regex=False)
            .str.replace(",", "", regex=False)
            .str.strip()
        )
    return pd.to_numeric(series, errors="coerce")


def clean_numeric_column(series: pd.Series) -> pd.Series:
    """Clean the column by removing commas and stripping whitespace."""
    if is_string_dtype(series) or is_object_dtype(series):  # Object or string type
        series = (
            series.astype(str)
            .str.replace(",", "", regex=False)
            .str.replace(
                r"[^0-9.-]", "", regex=True
            )  # Remove any non-numeric characters except dot and minus
            .str.strip()
        )
    return pd.to_numeric(series, errors="coerce")


def validate_column_exists(df: pd.DataFrame, target: str, logger) -> None:
    logger.info("Validating target column '%s' exists in DataFrame.", target)
    if target not in df.columns:
        logger.error("Target column '%s' not found. Available: %s", target, list(df.columns))
        raise ValueError(f"Target column '{target}' not found. Available: {list(df.columns)}")
    logger.info("'%s' column found", target)


def drop_rows_with_missing_target(df: pd.DataFrame, target: str, logger) -> pd.DataFrame:
    before_rows = df.shape[0]
    df = df.dropna(subset=[target])  # Drop rows where target is NaN
    dropped = before_rows - df.shape[0]
    logger.info("Dropped %s rows with missing %s values.", dropped, target)
    return df


def drop_rows_not_in_range(df: pd.DataFrame, target: str, min_val, max_val, logger) -> pd.DataFrame:
    before_rows = df.shape[0]
    df = df[df[target].between(min_val, max_val)]
    dropped = before_rows - df.shape[0]
    logger.info(
        "Dropped %s rows with %s values outside of [%s, %s].", dropped, target, min_val, max_val
    )
    return df


def drop_rows_with_negative_target(
    df: pd.DataFrame, target: str, logger, drop_zero: bool = False
) -> pd.DataFrame:
    before_rows = df.shape[0]
    if drop_zero:
        df = df[df[target] > 0]  # Drop rows where target is negative or zero
    else:
        df = df[df[target] >= 0]  # Drop rows where target is negative
    dropped = before_rows - df.shape[0]
    logger.info("Dropped %s rows with negative %s values.", dropped, target)
    return df


def drop_empty_columns(df: pd.DataFrame, logger) -> pd.DataFrame:
    logger.info("Dropping columns that are all NaN.")
    before_cols = df.shape[1]
    df = df.dropna(axis=1, how="all")  # Drop columns that are all NaN
    logger.info("Dropped %s empty columns.", before_cols - df.shape[1])
    return df


def remove_duplicate_rows(df: pd.DataFrame, logger) -> pd.DataFrame:
    logger.info("Dropping duplicate rows.")
    before_rows = df.shape[0]
    df = df.drop_duplicates()
    logger.info("Dropped %s duplicate rows.", before_rows - df.shape[0])
    return df


def remove_invalid_target_rows(df: pd.DataFrame, target: str, logger, config: dict) -> pd.DataFrame:
    logger.info("Removing rows with invalid target values.")
    before_rows = df.shape[0]
    if not before_rows:
        logger.warning("DataFrame is empty. No rows to remove.")
        return df
    df = df.dropna(subset=[target])
    logger.info("Dropped %s rows with NaN target values.", before_rows - df.shape[0])
    n_negative = (df[target] < 0).sum()
    if config["data"]["allow_zero_price"]:
        df = df[df[target] >= 0]
        logger.info("Dropped %s rows with negative target values.", n_negative)
    else:
        n_zero = (df[target] == 0).sum()
        df = df[df[target] > 0]
        logger.info("Dropped %s rows with negative target values.", n_negative)
        logger.info("Dropped %s rows with zero target values.", n_zero)

    logger.info("Dropped total %s rows with invalid target values.", before_rows - df.shape[0])
    dropped = before_rows - df.shape[0]
    if dropped / before_rows > config["data"]["invalid_target_threshold"]:
        logger.warning(
            "Dropped more than %s%% of rows due to invalid target values. Check your data and cleaning rules.",
            config["data"]["invalid_target_threshold"] * 100,
        )
    return df


def run_preprocess(in_path: Path, out_path: Path, target: str, logger, config: dict) -> None:
    in_path = Path(in_path)
    out_path = Path(out_path)

    df = read_data_from_file(in_path, logger)
    df_out = preprocess_data(df, target=target, logger=logger, config=config)
    save_preprocessed_data(df_out, out_path, logger)


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess used car dataset.")
    parser.add_argument("--in", dest="in_path", required=True, help="Input CSV path")
    parser.add_argument("--out", dest="out_path", required=True, help="Output CSV path")
    parser.add_argument("--target", default="price", help="Target column name")
    parser.add_argument(
        "--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"]
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/train.yaml"),
        help="Path to YAML config file with training parameters",
    )
    args = parser.parse_args()
    config = load_config(args.config)

    logger = setup_logging(level=args.log_level)

    run_preprocess(args.in_path, args.out_path, args.target, logger, config=config)


if __name__ == "__main__":
    main()
