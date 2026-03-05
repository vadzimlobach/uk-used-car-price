import json
import argparse
from pathlib import Path

import joblib
import pandas as pd
from pydantic import ValidationError
from src.schema import CarFeatures
from src.preprocess import add_features
from src.logging_config import setup_logging
from src.train import load_config

def load_json(path: Path) -> dict:
    with path.open('r', encoding='utf-8') as f:
        return json.load(f)
    
def main() -> None:
    parser = argparse.ArgumentParser(description='Run inference for UK used car price prediction.')
    parser.add_argument('--model', type=Path, required=True, help="Path to model.joblib")
    parser.add_argument('--input', type=Path, required=True, help="Path to input JSON")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/train.yaml"),
        help="Path to YAML config file with training parameters",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    logger = setup_logging(config['log_level'])

    raw = load_json(args.input)
    try:
        features = CarFeatures(**raw)
    except ValidationError as e:
        raise SystemExit(f"Input validation failed:\n{e}") from e

    model = joblib.load(args.model)
    X = pd.DataFrame([features.to_dict()])
    X_pred = add_features(X, logger, config['data'])
    if X_pred.shape[0] != 1:
        raise SystemExit(
            "Input was rejected by preprocessing rules (row dropped). "
            "Check year range, mileage >= 0, and mileage_per_year within allowed limits."
        )
    pred = model.predict(X_pred)[0]

    print(json.dumps({"predicted_price": float(pred)}, indent=2))
    

if __name__ == '__main__':
    main()