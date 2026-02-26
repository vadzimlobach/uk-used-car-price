from __future__ import annotations

import subprocess
import sys
from pathlib import Path
import pandas as pd


def test_preprocess_cli_creates_output_csv(tmp_path: Path):
    # Arrange: create input CSV
    in_path = tmp_path / "in.csv"
    out_path = tmp_path / "out.csv"

    df = pd.DataFrame({
        "price": ["£1000", "£2000", None, "-50"],
        "mileage": [10_000, None, 30_000, 40_000],
        "transmission": ["Manual", None, "Auto", "Manual"],
    })
    df.to_csv(in_path, index=False)

    # Act: run CLI module
    cmd = [
        sys.executable, "-m", "src.preprocess",
        "--in", str(in_path),
        "--out", str(out_path),
        "--target", "price",
        "--log-level", "ERROR",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)

    # Assert
    assert result.returncode == 0
    assert out_path.exists()

    out_df = pd.read_csv(out_path)
    assert "price" in out_df.columns
    assert (out_df["price"] >= 0).all()
