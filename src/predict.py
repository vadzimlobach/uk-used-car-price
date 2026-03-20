import argparse
import json
import os
from pathlib import Path

import joblib
import pandas as pd
from pydantic import ValidationError

from src.config import load_config
from src.contracts import SupportsPredict
from src.logging_config import setup_logging
from src.preprocess import add_features
from src.run_utils import resolve_latest_model_path
from src.schema import CarFeatures


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def resolve_model_path(cli_model: Path | None) -> Path:
    if cli_model is not None:
        if not cli_model.exists():
            raise SystemExit(f"Model not found at {cli_model}")
        return cli_model

    env_model = os.getenv("MODEL_PATH")
    if env_model:
        env_path = Path(env_model)
        if not env_path.exists():
            raise SystemExit(f"MODEL_PATH is set but file not found: {env_path}")
        return env_path

    return resolve_latest_model_path()


def predict_price(model: SupportsPredict, features: CarFeatures, logger, config: dict) -> float:
    """
    Run preprocessing and prediction for a single car example.
    """
    X = pd.DataFrame([features.to_dict()])
    X_pred = add_features(X, logger, config["data"])

    pred = model.predict(X_pred)[0]

    return float(pred)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run inference for UK used car price prediction.")
    parser.add_argument(
        "--model",
        type=Path,
        required=False,
        help="Path to model.joblib (optional — latest model used if not provided)",
    )
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

    model_path = resolve_model_path(args.model)
    logger.info("Loading model from %s", model_path)
    model = joblib.load(model_path)
    pred = predict_price(model, features, logger, config)

    print(json.dumps({"predicted_price": float(pred)}, indent=2))


if __name__ == "__main__":
    main()
