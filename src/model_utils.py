from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def build_model_pipeline(X_train: pd.DataFrame, config: dict) -> Pipeline:
    """Build preprocessing + regressor pipeline from config."""
    preprocessor = build_preprocessor(X_train)

    model_type = config["model_type"]
    random_state = int(config["random_state"])

    model_cfg = config["model"].get(model_type, {})
    regressor = build_regressor(model_type, model_cfg, random_state)

    return Pipeline(steps=[("preprocessor", preprocessor), ("regressor", regressor)])


def build_preprocessor(X_train: pd.DataFrame) -> ColumnTransformer:
    numeric_features = X_train.select_dtypes(include=["number"]).columns.tolist()
    categorical_features = X_train.select_dtypes(
        include=["object", "string", "category"]
    ).columns.tolist()

    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown="ignore", sparse_output=False)

    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ],
        remainder="drop",
    )


def build_regressor(model_type: str, model_config: dict, random_state: int) -> Any:
    params = (model_config or {}).get("params", {}) or {}

    if model_type == "linear":
        return LinearRegression(**params)

    if model_type == "rf":
        return RandomForestRegressor(random_state=random_state, **params)

    if model_type == "gb":
        return HistGradientBoostingRegressor(random_state=random_state, **params)

    raise ValueError(f"Unknown model_type: {model_type}")


def wrap_log_target(estimator: Any, log_target: bool) -> Any:
    """Optionally wrap estimator so it trains on log1p(y) but predicts in original scale."""
    if not log_target:
        return estimator

    return TransformedTargetRegressor(
        regressor=estimator,
        func=np.log1p,
        inverse_func=np.expm1,
        check_inverse=False,
    )
