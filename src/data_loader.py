import argparse
from pathlib import Path

import pandas as pd

from src.logging_config import setup_logging

TARGET_COLS = [
    "brand",
    "model",
    "year",
    "price",
    "transmission",
    "mileage",
    "fuelType",
    "tax",
    "mpg",
    "engineSize",
]

RENAME_MAP = {
    "tax(£)": "tax",
    "fuel type": "fuelType",
    "fuelType": "fuelType",
    "engine size": "engineSize",
    "engineSize": "engineSize",
}


def combine_csv_files(directory: str | Path, logger) -> pd.DataFrame:
    """Combine all CSV files in the given directory into a single DataFrame."""
    p = Path(directory)
    if not p.exists():
        logger.error("Directory not found: %s", p.resolve())
        raise FileNotFoundError(f"Directory {p.resolve()} does not exist.")
    if not p.is_dir():
        logger.error("Path is not a directory: %s", p.resolve())
        raise ValueError(f"Path {p.resolve()} is not a directory.")
    csv_files = sorted(p.glob("*.csv"))
    csv_files = [
        f for f in csv_files if f.name != "combined.csv"
    ]  # Exclude combined.csv if it exists
    if not csv_files:
        logger.error("No CSV files found in directory: %s", p.resolve())
        raise ValueError(f"No CSV files found in directory {p.resolve()}.")
    logger.info("Found %d CSV files in directory: %s", len(csv_files), p)
    combined_df = pd.DataFrame()
    base_columns = None
    total_rows = 0
    for csv_file in csv_files:
        logger.info("Loading CSV: %s", csv_file)
        df = pd.read_csv(csv_file)
        brand = csv_file.stem
        if brand.lower().__contains__("cclass") or brand.lower().__contains__("merc"):
            brand = "mercedes"
        if brand.lower().__contains__("focus"):
            brand = "ford"
        df["brand"] = brand  # Add brand column based on filename
        logger.info("number of rows in %s: %s, columns: %s", csv_file, df.shape[0], df.shape[1])
        df = standardize_columns(df, logger)
        if base_columns is None:
            base_columns = set(df.columns)
        else:
            if set(df.columns) != base_columns:
                logger.warning("Column mismatch in %s", csv_file)
        combined_df = pd.concat([combined_df, df], ignore_index=True)
        total_rows += df.shape[0]
    logger.info("Combined DataFrame total rows before dropping all-NaN columns: %s", total_rows)
    combined_df = combined_df.dropna(how="all")  # Drop columns that are all NaN
    logger.info("Combined DataFrame shape: %s", combined_df.shape)
    logger.info("Saving combined CSV to: %s", p / "combined.csv")
    return combined_df


def standardize_columns(df: pd.DataFrame, logger) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]

    df = df.rename(columns=RENAME_MAP)

    COALESCE_PAIRS = [
        ("mileage", "mileage2"),
        ("fuelType", "fuelType2"),
        ("engineSize", "engineSize2"),
    ]

    for primary, secondary in COALESCE_PAIRS:
        df = coalesce_columns(df, primary, secondary, logger)

    missing = [c for c in TARGET_COLS if c not in df.columns]
    if missing:
        logger.warning("Missing expected columns after standardization: %s", missing)
        for c in missing:
            df[c] = pd.NA  # create missing columns

    df = df[TARGET_COLS]
    return df


def coalesce_columns(
    df: pd.DataFrame,
    primary: str,
    secondary: str,
    logger,
) -> pd.DataFrame:
    """
    Merge secondary column into primary column.
    Primary takes precedence; secondary fills missing values.
    """
    if secondary not in df.columns:
        return df

    logger.info("Coalescing '%s' <- '%s'", primary, secondary)

    if primary not in df.columns:
        df[primary] = pd.NA

    before_missing = df[primary].isna().sum()

    df[primary] = df[primary].fillna(df[secondary])

    after_missing = df[primary].isna().sum()
    filled = before_missing - after_missing

    logger.info("Filled %s missing values in '%s' from '%s'", filled, primary, secondary)

    return df


def basic_report(df: pd.DataFrame, target: str, logger) -> None:
    """Print a basic report of the data."""
    logger.info("===DATA REPORT===")
    logger.info(f"Number of rows: {len(df):,}")
    logger.info("Number of columns: %s", df.shape[1])
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
            tgt.astype("str")
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


def load_data(path: str | Path, logger) -> pd.DataFrame:
    """Load the data from the given directory.

    Args:
        data_dir (Path): The directory containing the data."""
    p = Path(path)
    if not p.exists():
        logger.error("CSV not found: %s", p.resolve())
        raise FileNotFoundError(f"Path {p.resolve()} does not exist.")
    if not p.is_file():
        logger.error("Path is not a file: %s", p.resolve())
        raise ValueError(f"Path {p.resolve()} is not a file.")
    if p.suffix != ".csv":
        logger.error("Path is not a CSV file: %s", p.resolve())
        raise ValueError(f"Path {p.resolve()} is not a CSV file.")
    logger.info("Loading CSV: %s", p)
    return pd.read_csv(p)


def main() -> None:
    parser = argparse.ArgumentParser(description="Load CSV and print a basic data quality report.")
    parser.add_argument("--path", required=True, help="Path to CSV in data/raw/")
    parser.add_argument("--target", default="price", help="Target column name (default: price)")
    parser.add_argument(
        "--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"]
    )
    parser.add_argument(
        "--log-file", default=None, help="Optional log file path, e.g. logs/run.log"
    )
    args = parser.parse_args()

    logger = setup_logging(level=args.log_level, log_file=args.log_file)

    try:
        df = combine_csv_files(args.path, logger)
        logger.info("Saving combined CSV to: %s", args.path + "/combined.csv")
        df.to_csv(Path(args.path + "/combined.csv"), index=False)
        # df = load_data(args.path, logger)
        basic_report(df, args.target, logger)
    except Exception:
        # Logs full stack trace
        logger.exception("Fatal error running data report")
        raise


if __name__ == "__main__":
    main()
