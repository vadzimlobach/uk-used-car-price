import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Protocol

import joblib
import pandas as pd
from pydantic import ValidationError

from src.logging_config import setup_logging
from src.preprocess import add_features
from src.schema import CarFeatures
from src.train import load_config


class SupportsPredict(Protocol):
    def predict(self, X: pd.DataFrame) -> Sequence[float]: ...


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def predict_price(model: SupportsPredict, features: CarFeatures, logger, config: dict) -> float:
    """
    Run preprocessing and prediction for a single car example.
    """
    X = pd.DataFrame([features.to_dict()])
    X_pred = add_features(X, logger, config)

    pred = model.predict(X_pred)[0]

    return float(pred)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run inference for UK used car price prediction.")
    parser.add_argument("--model", type=Path, required=True, help="Path to model.joblib")
    parser.add_argument("--input", type=Path, required=True, help="Path to input JSON")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/train.yaml"),
        help="Path to YAML config file with training parameters",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    logger = setup_logging(config["log_level"])

    raw = load_json(args.input)
    try:
        features = CarFeatures(**raw)
    except ValidationError as e:
        raise SystemExit(f"Input validation failed:\n{e}") from e

    model = joblib.load(args.model)
    pred = predict_price(model, features, logger, config)

    print(json.dumps({"predicted_price": float(pred)}, indent=2))


if __name__ == "__main__":
    main()
