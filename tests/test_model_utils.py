import numpy as np
import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline

from src.model_utils import (
    build_model_pipeline,
    build_preprocessor,
    build_regressor,
    wrap_log_target,
)


def _toy_df(n: int = 30) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    makes = np.array(["Ford", "VW", "BMW", "Audi"])
    return pd.DataFrame(
        {
            "price": rng.integers(1000, 20000, size=n),
            "year": rng.integers(2010, 2023, size=n),
            "mileage": rng.integers(5_000, 120_000, size=n),
            "make": rng.choice(makes, size=n),
        }
    )


def test_build_preprocessor_returns_column_transformer():
    X = _toy_df()
    pre = build_preprocessor(X)

    assert isinstance(pre, ColumnTransformer)

    # Fit required before feature names are available
    pre.fit(X)

    feature_names = pre.get_feature_names_out()

    # Expect both numeric and categorical features
    assert any("year" in f for f in feature_names)
    assert any("make" in f for f in feature_names)


def test_build_regressor_rf_applies_params_and_random_state():
    model_cfg = {"params": {"n_estimators": 10, "max_depth": 3}}
    reg = build_regressor("rf", model_cfg, random_state=123)

    assert isinstance(reg, RandomForestRegressor)

    params = reg.get_params(deep=False)
    assert params["n_estimators"] == 10
    assert params["max_depth"] == 3
    assert params["random_state"] == 123


def test_build_regressor_unknown_raises():
    with pytest.raises(ValueError):
        build_regressor("unknown", {"params": {}}, random_state=42)


def test_wrap_log_target_true_returns_transformed_target_regressor():
    X = _toy_df()
    config = {
        "model_type": "rf",
        "random_state": 42,
        "model": {"rf": {"params": {"n_estimators": 5}}},
    }
    pipe = build_model_pipeline(X, config)

    wrapped = wrap_log_target(pipe, log_target=True)
    assert isinstance(wrapped, TransformedTargetRegressor)


def test_wrap_log_target_false_returns_same_estimator():
    X = _toy_df()
    config = {
        "model_type": "rf",
        "random_state": 42,
        "model": {"rf": {"params": {"n_estimators": 5}}},
    }
    pipe = build_model_pipeline(X, config)

    wrapped = wrap_log_target(pipe, log_target=False)
    assert wrapped is pipe


def test_build_model_pipeline_is_pipeline_with_expected_steps():
    X = _toy_df()
    config = {
        "model_type": "rf",
        "random_state": 42,
        "model": {"rf": {"params": {"n_estimators": 5}}},
    }

    pipe = build_model_pipeline(X, config)
    assert isinstance(pipe, Pipeline)

    assert "preprocessor" in pipe.named_steps
    assert "regressor" in pipe.named_steps
