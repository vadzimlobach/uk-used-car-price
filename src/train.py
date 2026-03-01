import argparse
import json
import joblib
from datetime import datetime, timezone
from typing import Any
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.model_selection import cross_validate, KFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, root_mean_squared_error, mean_absolute_error, make_scorer
import pandas as pd
import numpy as np
from pathlib import Path
from src.data_io import read_data_from_file
from src.logging_config import setup_logging

random_state = 42

def train_test_split_data(in_path: Path, logger) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = read_data_from_file(in_path, logger)
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=random_state)
    return train_df, test_df, 

def set_X_y(train_df: pd.DataFrame, test_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    X_train = train_df.drop(columns=["price"])
    y_train = train_df["price"]
    X_test = test_df.drop(columns=["price"])
    y_test = test_df["price"]
    return X_train, y_train, X_test, y_test

def build_model_pipeline(X_train: pd.DataFrame, model_type: str="linear") -> Pipeline:
    numeric_features=X_train.select_dtypes(include=["number"]).columns
    categorical_features = X_train.select_dtypes(include=["object", "string", "category"]).columns

    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown="ignore", sparse_output=False)

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, list(numeric_features)),
            ("cat", categorical_transformer, list(categorical_features))
        ], remainder="drop"
    )

    if model_type == "linear":
        regressor = LinearRegression()
    elif model_type == "rf":
        regressor = RandomForestRegressor(
            n_estimators=300,
            random_state=42,
            n_jobs=-1,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type}")


    model = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("regressor", regressor),
    ])

    return model

def train_and_evaluate_model(model: Pipeline, X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.Series, log_target: bool=False) -> dict:
    if log_target:
        y_train_fit = np.log1p(y_train)
        model.fit(X_train, y_train_fit)

        y_hat_log = model.predict(X_test)
        y_hat = np.expm1(y_hat_log)  # back to pounds
    else:
        model.fit(X_train, y_train)
        y_hat = model.predict(X_test)

    # Safety: avoid negative predictions causing weird metrics interpretation
    y_hat = np.maximum(y_hat, 0)

    rmse = root_mean_squared_error(y_test, y_hat)
    mae = mean_absolute_error(y_test, y_hat)
    r2 = r2_score(y_test, y_hat)
    return {
        "rmse": rmse,
        "mae": mae,
        "r2": r2
    }

def save_artifacts(
    *,
    model: Pipeline,
    metrics: dict[str, Any],
    model_out: Path,
    metrics_out: Path,
    logger,
) -> None:
    model_out.parent.mkdir(parents=True, exist_ok=True)
    metrics_out.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, model_out)
    logger.info("Saved model to: %s", model_out)

    with metrics_out.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, sort_keys=True)
    logger.info("Saved metrics to: %s", metrics_out)

def rmse(y_true, y_pred) -> float:
    return root_mean_squared_error(y_true, y_pred)

def cross_validate_model(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    base_pipeline: Pipeline,
    n_splits: int = 5,
    random_state: int = 42,
    log_target: bool = False,
) -> dict:
    """
    Cross-validate the model.
    If log_target=True, wraps model in TransformedTargetRegressor using log1p/expm1.
    Returns mean/std for RMSE, MAE, and R2.
    """
    if log_target:
        estimator = TransformedTargetRegressor(
            regressor=base_pipeline,
            func=np.log1p,
            inverse_func=np.expm1,
            check_inverse=False,  # safe for log1p/expm1
        )
    else:
        estimator = base_pipeline

    cv = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    scoring = {
        "rmse": make_scorer(rmse, greater_is_better=False),  # negative because sklearn expects "higher is better"
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

    # Convert negative scores back to positive errors
    rmse_scores = -results["test_rmse"]
    mae_scores = -results["test_mae"]
    r2_scores = results["test_r2"]

    summary = {
        "rmse_mean": float(rmse_scores.mean()),
        "rmse_std": float(rmse_scores.std(ddof=1)),
        "mae_mean": float(mae_scores.mean()),
        "mae_std": float(mae_scores.std(ddof=1)),
        "r2_mean": float(r2_scores.mean()),
        "r2_std": float(r2_scores.std(ddof=1)),
        "n_splits": n_splits,
        "log_target": bool(log_target),
    }
    return summary

def main() -> None:
    parser = argparse.ArgumentParser(description="Train a model to predict used car prices.")
    parser.add_argument("--log-level", default="INFO", help="Set logging level")
    parser.add_argument("--in", dest="in_path", required=True, help="Input CSV path")
    parser.add_argument("--model-out", dest="model_out_path", default='src/models/baseline.joblib', help="Output path for the trained model (optional)")
    parser.add_argument("--metrics-out", dest="metrics_out_path", default='src/reports/baseline_metrics.json', help="Output path for the evaluation metrics (optional)")
    parser.add_argument("--run-name", dest="run_name", default="baseline_linear_v1", help="Name for the training run (optional)")
    parser.add_argument("--target", default="price", help="Target column name")
    parser.add_argument("--log-target", dest="log_target", action="store_true", help="Train on log1p(target) and invert predictions for metric reporting")
    parser.add_argument("--model", default="linear", choices=["linear", "rf"], help="Which regressor to train")
    args = parser.parse_args()

    logger = setup_logging(level=args.log_level)
    in_path = Path(args.in_path)
    model_out = Path(args.model_out_path)
    metrics_out = Path(args.metrics_out_path)
    run_name = args.run_name

    if args.log_target:
    # only auto-adjust if user left defaults unchanged
        if args.model_out_path == "src/models/baseline.joblib":
            model_out = Path("src/models/baseline_logtarget.joblib")
        if args.metrics_out_path == "src/reports/baseline_metrics.json":
            metrics_out = Path("src/reports/baseline_metrics_logtarget.json")
        if args.run_name == "baseline_linear_v1":
            run_name = "baseline_linear_logtarget_v1"

    logger.info("Starting training with input data from %s", args.in_path)
    train_df, test_df = train_test_split_data(in_path=Path(args.in_path), logger=logger)
    X_train, y_train, X_test, y_test = set_X_y(train_df, test_df)
    model = build_model_pipeline(X_train, model_type=args.model)
    cv_model = build_model_pipeline(X_train, model_type=args.model)  # separate instance for CV to avoid data leakage from fitting the final model
    cv_summary = cross_validate_model(
    X_train,
    y_train,
    base_pipeline=cv_model,
    n_splits=5,
    random_state=random_state,
    log_target=args.log_target,
)

    logger.info("CV summary: %s", cv_summary)
    metrics = train_and_evaluate_model(model, X_train, y_train, X_test, y_test, args.log_target)
    metrics_payload: dict[str, Any] = {
        "log-target": args.log_target,
        "run_name": run_name,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "input_path": str(in_path),
        "target": args.target,
        "random_state": random_state,
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "metrics": metrics,
        "model": {
            "pipeline": "ColumnTransformer + LinearRegression",
            "regressor": type(model.named_steps["regressor"]).__name__,
        },
    }
    save_artifacts(model=model, metrics=metrics_payload, model_out=model_out, metrics_out=metrics_out, logger=logger)
    logger.info("Model evaluation metrics: %s", metrics)

if __name__ == "__main__":
    main()