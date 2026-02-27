import argparse
from pathlib import Path
from venv import logger
import pandas as pd
from pandas.api.types import is_string_dtype
from src.logging_config import setup_logging
from src.data_io import read_data_from_file, save_preprocessed_data

PREPROCESS_CONFIG = {
    "allow_zero_price": False,
    "categorical_fill_value": "Unknown",
    "numeric_fill_strategy": "median",  # or "mean"
    "invalid_target_threshold": 0.1,  # Warn if more than 10% of rows are dropped due to invalid target
}

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

def preprocess_data(df: pd.DataFrame, target: str, logger, config: dict) -> pd.DataFrame:
    """Preprocess the data by cleaning the target column."""
    
    validate_target_exists(df, target, logger)
    
    logger.info("Starting preprocess. Raw shape=%s", df.shape)

    df = drop_empty_columns(df, logger)
    df = remove_duplicate_rows(df, logger)
    # Clean the target column
    df[target] = clean_price(df[target])
    df = remove_invalid_target_rows(df, target, logger, config)
    df = handle_missing_values(df, target, logger, config)
    
    logger.info("Finished preprocess. Cleaned shape=%s", df.shape)

    return df

def handle_missing_values(df: pd.DataFrame, target: str, logger, config: dict) -> pd.DataFrame:
    logger.info("Handling missing values.")
    df = handle_missing_num(df, target, logger, config)
    df = handle_missing_cat(df, logger, config)
    logger.info("Done handling missing values.")
    return df
        

def handle_missing_num(df: pd.DataFrame, target: str, logger, config: dict) -> pd.DataFrame:
    logger.info("Filling missing values in numeric columns with '%s'.", config["numeric_fill_strategy"])
    num_cols = df.select_dtypes(include=["number"]).columns
    for col in num_cols:
        if col == target:
            continue  # Skip target column
        if df[col].isna().any():
            replacement_val = calculate_num_replacement_value(df[col], config["numeric_fill_strategy"], logger)
            df[col] = df[col].fillna(replacement_val)
            logger.info("Filled missing values in numeric column '%s' with %s: %s", col, config["numeric_fill_strategy"], replacement_val)
    logger.info("Done filling numeric missing values.")
    return df

def calculate_num_replacement_value(series: pd.Series, strategy: str, logger) -> float:
    if strategy == "median":
        replacement = series.median()
    elif strategy == "mean":
        replacement = series.mean()
    else:
        logger.error("Invalid numeric fill strategy: %s", strategy)
        raise ValueError(f"Invalid numeric fill strategy: {strategy}")
    logger.debug("Calculated replacement value for numeric column: %s", replacement)
    return replacement

def handle_missing_cat(df: pd.DataFrame, logger, config: dict) -> pd.DataFrame:
    logger.info("Filling missing values in categorical columns with '%s'.", config["categorical_fill_value"])
    cat_cols = df.select_dtypes(include=["object", "string", "category"]).columns
    for col in cat_cols:
        if df[col].isna().any():
            df[col] = df[col].fillna(config["categorical_fill_value"])
            logger.info("Filled missing values in categorical column '%s' with '%s'", col, config["categorical_fill_value"])
    logger.info("Done filling categorical missing values.")
    return df


def validate_target_exists(df: pd.DataFrame, target: str, logger) -> None:
    logger.info("Validating target column '%s' exists in DataFrame.", target)
    if target not in df.columns:
        logger.error("Target column '%s' not found. Available: %s", target, list(df.columns))
        raise ValueError(f"Target column '{target}' not found. Available: {list(df.columns)}")
    logger.info("'%s' column found", target) 

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
    if config["allow_zero_price"]:
        df = df[df[target] >= 0]
        logger.info("Dropped %s rows with negative target values.", n_negative)
    else:
        n_zero = (df[target] == 0).sum()
        df = df[df[target] > 0]
        logger.info("Dropped %s rows with negative target values.", n_negative)
        logger.info("Dropped %s rows with zero target values.", n_zero)

    logger.info("Dropped total %s rows with invalid target values.", before_rows - df.shape[0])
    dropped= before_rows - df.shape[0]
    if dropped / before_rows > config["invalid_target_threshold"]:
        logger.warning("Dropped more than %s%% of rows due to invalid target values. Check your data and cleaning rules.", config["invalid_target_threshold"] * 100)
    return df
    
def run_preprocess(in_path: Path, out_path: Path, target: str, logger, config: dict = PREPROCESS_CONFIG) -> None:
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
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args()

    logger = setup_logging(level=args.log_level)

    run_preprocess(args.in_path, args.out_path, args.target, logger, config=PREPROCESS_CONFIG)

    


if __name__ == "__main__":
    main()