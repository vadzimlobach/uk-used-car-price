import argparse
import json
import joblib
import yaml
import pandas as pd
import numpy as np
from datetime import datetime, timezone
from typing import Any
from pathlib import Path

from sklearn.model_selection import train_test_split, cross_validate, KFold
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, root_mean_squared_error, mean_absolute_error, make_scorer

from src.data_io import read_data_from_file
from src.logging_config import setup_logging
from src.analyze import analyze_residuals
from src.model_utils import build_model_pipeline, wrap_log_target


def load_config(config_path: Path) -> dict:
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# -------------------------
# Data utils
# -------------------------
def train_test_split_data(in_path: Path, logger, random_state: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = read_data_from_file(in_path, logger)
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=random_state)
    return train_df, test_df


def set_X_y(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    *,
    target: str,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    if target not in train_df.columns:
        raise ValueError(f"Target column '{target}' not found. Available columns: {list(train_df.columns)}")

    X_train = train_df.drop(columns=[target])
    y_train = train_df[target]
    X_test = test_df.drop(columns=[target])
    y_test = test_df[target]
    return X_train, y_train, X_test, y_test


# -------------------------
# Training
# -------------------------
def train_model(base_model: Pipeline, X_train: pd.DataFrame, y_train: pd.Series, log_target: bool):
    estimator = wrap_log_target(base_model, log_target)
    estimator.fit(X_train, y_train)
    return estimator

# -------------------------
# Evaluation utils
# -------------------------
def rmse(y_true, y_pred) -> float:
    return root_mean_squared_error(y_true, y_pred)


def evaluate_on_holdout(estimator: Any, X_test: pd.DataFrame, y_test: pd.Series) -> dict[str, float]:
    y_hat = estimator.predict(X_test)

    # Safety: avoid negative predictions causing weird metrics interpretation
    y_hat = np.maximum(y_hat, 0)

    return {
        "rmse": float(root_mean_squared_error(y_test, y_hat)),
        "mae": float(mean_absolute_error(y_test, y_hat)),
        "r2": float(r2_score(y_test, y_hat)),
    }


def cross_validate_model(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    base_pipeline: Pipeline,
    n_splits: int,
    random_state: int,
    log_target: bool,
    shuffle: bool,
) -> dict:
    """Cross-validate the model, optionally using log1p/expm1 target transform."""
    estimator = wrap_log_target(base_pipeline, log_target)

    cv = KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)

    scoring = {
        "rmse": make_scorer(rmse, greater_is_better=False),
        "mae": make_scorer(mean_absolute_error, greater_is_better=False),
        "r2": "r2",
    }

    results = cross_validate(
        estimator,
        X,
        y,
        cv=cv,
        scoring=scoring,
        n_jobs=-1,
        return_train_score=False,
    )

    rmse_scores = -results["test_rmse"]
    mae_scores = -results["test_mae"]
    r2_scores = results["test_r2"]

    return {
        "rmse_mean": float(rmse_scores.mean()),
        "rmse_std": float(rmse_scores.std(ddof=1)),
        "mae_mean": float(mae_scores.mean()),
        "mae_std": float(mae_scores.std(ddof=1)),
        "r2_mean": float(r2_scores.mean()),
        "r2_std": float(r2_scores.std(ddof=1)),
        "n_splits": int(n_splits),
        "shuffle": bool(shuffle),
        "log_target": bool(log_target),
    }


# -------------------------
# Artifacts
# -------------------------
def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def save_artifacts(
    *,
    model: Any,
    metrics: dict[str, Any],
    model_out: Path,
    metrics_out: Path,
    logger,
) -> None:
    model_out.parent.mkdir(parents=True, exist_ok=True)
    metrics_out.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, model_out)
    logger.info("Saved model to: %s", model_out)

    write_json(metrics_out, metrics)
    logger.info("Saved metrics to: %s", metrics_out)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a model to predict used car prices (config-driven).")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/train.yaml"),
        help="Path to YAML config file with training parameters",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    logger = setup_logging(config["log_level"])

    run_name = config.get("run_name", "run")
    random_state = int(config["random_state"])
    input_path = Path(config["data"]["input_path"])
    target = config["data"]["target"]

    cv_config = config["cv"]
    model_path = Path(config["output"]["model_path"])
    metrics_path = Path(config["output"]["metrics_path"])
    cv_summary_path = Path(config["output"]["cv_summary_path"])

    model_type = config["model_type"]
    log_target = bool(config["model"].get("log_target", False))

    logger.info("Run name: %s", run_name)
    logger.info("Starting training with input data from %s", input_path)

    train_df, test_df = train_test_split_data(in_path=input_path, logger=logger, random_state=random_state)
    X_train, y_train, X_test, y_test = set_X_y(train_df, test_df, target=target)

    # Build a fresh pipeline instance for CV and for final training.
    base_model = build_model_pipeline(X_train, config)
    cv_model = build_model_pipeline(X_train, config)

    cv_summary = cross_validate_model(
        X_train,
        y_train,
        base_pipeline=cv_model,
        n_splits=int(cv_config["n_splits"]),
        random_state=random_state,
        log_target=log_target,
        shuffle=bool(cv_config.get("shuffle", True)),
    )
    logger.info("CV summary: %s", cv_summary)

    # Save CV summary as a first-class artifact.
    write_json(
        cv_summary_path,
        {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "run_name": run_name,
            "model_type": model_type,
            "cv": cv_summary,
            "config": config,
        },
    )
    logger.info("Saved CV summary to: %s", cv_summary_path)

    # Final estimator: wrap for log_target so saved model predicts in original scale.
    final_estimator = train_model(base_model, X_train, y_train, log_target)  # type: ignore

    metrics = evaluate_on_holdout(final_estimator, X_test, y_test)

    # Residual analysis (best-effort)
    try:
        analyze_residuals(final_estimator, X_test, y_test, logger)
    except Exception as e:
        logger.warning("Residual analysis failed (non-fatal): %s", e)

    metrics_payload: dict[str, Any] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "run_name": run_name,
        "data": {
            "input_path": str(input_path),
            "target": target,
            "train_rows": int(len(train_df)),
            "test_rows": int(len(test_df)),
        },
        "model": {
            "model_type": model_type,
            "log_target": log_target,
        },
        "cv": cv_summary,
        "metrics": metrics,
        "config": config,
    }

    save_artifacts(model=final_estimator, metrics=metrics_payload, model_out=model_path, metrics_out=metrics_path, logger=logger)
    logger.info("Model evaluation metrics: %s", metrics)


if __name__ == "__main__":
    main()
