import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, cast

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException

from src.config import load_config
from src.contracts import HealthResponse, PredictionResponse, SupportsPredict
from src.logging_config import setup_logging
from src.preprocess import add_features
from src.run_utils import resolve_latest_model_path
from src.schema import CarFeatures


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:

    model_path_env = os.getenv("MODEL_PATH")
    model_path = Path(model_path_env) if model_path_env else resolve_latest_model_path()

    config = load_config(Path("configs/train.yaml"))
    logger = setup_logging(config["log_level"])

    logger.info("Loading model from %s", model_path)

    model = cast(SupportsPredict, joblib.load(model_path))

    app.state.config = config
    app.state.logger = logger
    app.state.model = model
    app.state.model_path = model_path

    yield

    logger.info("Shutting down API")


app = FastAPI(
    title="UK Used Car Price Predictor",
    lifespan=lifespan,
)


def get_model() -> SupportsPredict:
    return cast(SupportsPredict, app.state.model)


def get_logger() -> Any:
    return app.state.logger


def get_config() -> dict[str, Any]:
    return cast(dict[str, Any], app.state.config)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    if not hasattr(app.state, "model"):
        raise HTTPException(status_code=503, detail="Model not loaded")
    return HealthResponse(status="OK")


@app.post("/predict", response_model=PredictionResponse)
def predict(features: CarFeatures) -> PredictionResponse:
    config = get_config()
    model = get_model()
    logger = get_logger()
    try:
        X = pd.DataFrame([features.to_dict()])
        X = add_features(X, logger, config["data"])

        pred = float(model.predict(X)[0])

        return PredictionResponse(
            predicted_price=pred, model_run_id=app.state.model_path.parent.name
        )

    except Exception as exc:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail="Prediction failed") from exc
