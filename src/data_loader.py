import argparse
from pathlib import Path
import pandas as pd

from src.logging_config import setup_logging

def combine_csv_files(directory: str|Path, logger) -> pd.DataFrame:
    """Combine all CSV files in the given directory into a single DataFrame."""
    p = Path(directory)
    if not p.exists():
        logger.error("Directory not found: %s",  p.resolve())
        raise FileNotFoundError(f"Directory {p.resolve()} does not exist.")
    if not p.is_dir():
        logger.error("Path is not a directory: %s",  p.resolve())
        raise ValueError(f"Path {p.resolve()} is not a directory.") 
    csv_files = list(p.glob("*.csv"))
    if not csv_files:
        logger.error("No CSV files found in directory: %s",  p.resolve())
        raise ValueError(f"No CSV files found in directory {p.resolve()}.")
    logger.info("Found %d CSV files in directory: %s", len(csv_files), p)
    df_list = []
    for csv_file in csv_files:
        logger.info("Loading CSV: %s", csv_file)
        df = pd.read_csv(csv_file)
        df_list.append(df)
    combined_df = pd.concat(df_list, ignore_index=True)
    logger.info("Combined DataFrame shape: %s", combined_df.shape)
    logger.info("Saving combined CSV to: %s", p / "combined.csv")
    combined_df.to_csv(p / "combined.csv", index=False)
    return combined_df

def load_data(path: str|Path, logger) -> pd.DataFrame:
    """Load the data from the given directory.

    Args:
        data_dir (Path): The directory containing the data."""
    p = Path(path)
    if not p.exists():
        logger.error("CSV not found: %s",  p.resolve())
        raise FileNotFoundError(f"Path {p.resolve()} does not exist.")
    if not p.is_file():
        logger.error("Path is not a file: %s",  p.resolve())
        raise ValueError(f"Path {p.resolve()} is not a file.")
    if p.suffix != ".csv":
        logger.error("Path is not a CSV file: %s",  p.resolve())
        raise ValueError(f"Path {p.resolve()} is not a CSV file.")
    logger.info("Loading CSV: %s", p)
    return pd.read_csv(p)

def basic_report(df: pd.DataFrame, target: str, logger) -> None:
    """Print a basic report of the data."""
    logger.info("===DATA REPORT===")
    logger.info(f"Number of rows: {len(df):,}")
    logger.info(f"Number of columns: %s", {df.shape[1]})
    logger.info(f"Shape: {df.shape}")
    logger.info("Columns: %s", ", ".join(df.columns))

    logger.info("DTypes: %s", df.dtypes.astype(str).sort_values())

    logger.info("Missing values:")
    logger.info(df.isna().sum().sort_values(ascending=False))

    if target not in df.columns:
        logger.error("Target column '%s' not found. Available: %s", target, list(df.columns))
        raise ValueError(f"Target column '{target}' not found. Available: {list(df.columns)}") 
    
    # Target column summary
    tgt = df[target]
    logger.info(f"Target '{target}' summary:")
    logger.info(tgt.describe(include="all"))
 
    # Coerce target if is object
    if tgt.dtype == "object":
        coerced = (
            tgt.astype('str')
            .str.strip()
            .str.replace(",", "", regex=False)
            .str.replace("£", "", regex=False)
        )
        tgt_num = pd.to_numeric(coerced, errors="coerce")
        n_bad = tgt_num.isna().sum()
        logger.info(f"Target coercion attempt: numeric NaNs after coercion: {n_bad:,}")
        logger.debug(tgt_num.describe())
    
    # Basic duplicates check
    n_dups = df.duplicated().sum()
    logger.info(f"Number of duplicate rows: {n_dups:,}")

    logger.info("===END OF REPORT===")

def main() -> None:
    parser = argparse.ArgumentParser(description="Load CSV and print a basic data quality report.")
    parser.add_argument("--path", required=True, help="Path to CSV in data/raw/")
    parser.add_argument("--target", default="price", help="Target column name (default: price)")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    parser.add_argument("--log-file", default=None, help="Optional log file path, e.g. logs/run.log")
    args = parser.parse_args()

    logger = setup_logging(level=args.log_level, log_file=args.log_file)

    try:
        #df = combine_csv_files(args.path, logger)
        df = load_data(args.path, logger)
        basic_report(df, args.target, logger)
    except Exception:
        # Logs full stack trace
        logger.exception("Fatal error running data report")
        raise


if __name__ == "__main__":
    main()