
from pathlib import Path
import pandas as pd


def read_data_from_file(in_path: Path, logger) -> pd.DataFrame:
    logger.info("Loading CSV: %s", in_path)
    df = pd.read_csv(in_path)
    return df

def save_preprocessed_data(df: pd.DataFrame, out_path: Path, logger) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Saving processed CSV: %s", out_path)
    df.to_csv(out_path, index=False)
    logger.info("Done.")