import argparse
from pathlib import Path
import pandas as pd
from pandas.api.types import is_string_dtype

from src.logging_config import setup_logging

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

def preprocess_data(df: pd.DataFrame, target: str, logger) -> pd.DataFrame:
    """Preprocess the data by cleaning the target column."""
    if target not in df.columns:
        logger.error("Target column '%s' not found. Available: %s", target, list(df.columns))
        raise ValueError(f"Target column '{target}' not found. Available: {list(df.columns)}") 
    
    logger.info("Starting preprocess. Raw shape=%s", df.shape)

    # Drop empty columns
    before_cols = df.shape[1]
    df = df.dropna(axis=1, how="all")  # Drop columns that are all NaN
    logger.info("Dropped %s empty columns.", before_cols - df.shape[1])

    # Remove duplicate rows
    before_rows = df.shape[0]
    df = df.drop_duplicates()
    logger.info("Dropped %s duplicate rows.", before_rows - df.shape[0])

    # Clean the target column
    df[target] = clean_price(df[target])

    # remove rows where target is missing or invalid after coercion
    before_rows = df.shape[0]
    df = df.dropna(subset=[target])
    df = df[df[target] >= 0]  # Remove rows with negative prices
    logger.info("Dropped %s rows with invalid target values.", before_rows - df.shape[0])

    # Missing value handling for numeric columns (except target): fill with median
    num_cols = df.select_dtypes(include=["number"]).columns
    for col in num_cols:
        if col == target:
            continue  # Skip target column
        if df[col].isna().any():
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
            logger.info("Filled missing values in numeric column '%s' with median: %s", col, median_val)
    
    # Missing value handling for categorical columns: fill with Unknown
    cat_cols = df.select_dtypes(include=["object", "string"]).columns
    for col in cat_cols:
        if df[col].isna().any():
            df[col] = df[col].fillna("Unknown")
            logger.info("Filled missing values in categorical column '%s' with 'Unknown'", col)
    
    logger.info("Finished preprocess. Cleaned shape=%s", df.shape)

    return df

def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess used car dataset.")
    parser.add_argument("--in", dest="in_path", required=True, help="Input CSV path")
    parser.add_argument("--out", dest="out_path", required=True, help="Output CSV path")
    parser.add_argument("--target", default="price", help="Target column name")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args()

    logger = setup_logging(level=args.log_level)

    in_path = Path(args.in_path)
    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Loading: %s", in_path)
    df = pd.read_csv(in_path)

    df_out = preprocess_data(df, target=args.target, logger=logger)

    logger.info("Saving processed CSV: %s", out_path)
    df_out.to_csv(out_path, index=False)
    logger.info("Done.")


if __name__ == "__main__":
    main()