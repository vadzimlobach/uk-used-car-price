import pandas as pd
from pathlib import Path

def helper_function(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df.groupby("transmission")["price"].describe()

def main():
    print(helper_function(Path("data/processed/processed.csv")))

if __name__ == "__main__":
    main()
    