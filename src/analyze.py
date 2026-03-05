import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def analyze_residuals(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    logger,
    out_dir: str | Path | None = None,
    prefix: str = "residuals",
) -> None:
    """
    Residual diagnostics (saved to disk when out_dir is provided):
    - Residual vs predicted
    - Residual vs true price
    - Residual vs car_age
    - Residual vs mileage_per_year
    - Residual distribution
    """

    # Predictions
    y_pred = model.predict(X_test)
    y_pred = np.maximum(y_pred, 0)

    residuals = y_test - y_pred

    logger.info("Residual summary:")
    logger.info(residuals.describe())

    out_path = Path(out_dir) if out_dir is not None else None
    if out_path is not None:
        out_path.mkdir(parents=True, exist_ok=True)

    def _save_or_close(fig, name: str):
        if out_path is not None:
            fig.savefig(out_path / f"{prefix}_{name}.png", bbox_inches="tight", dpi=150)
        plt.close(fig)

    # Residual vs Predicted
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(y_pred, residuals, alpha=0.3)
    ax.axhline(0)
    ax.set_xlabel("Predicted price")
    ax.set_ylabel("Residual (y_true - y_pred)")
    ax.set_title("Residual vs Predicted")
    _save_or_close(fig, "vs_predicted")

    # Residual vs True
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(y_test, residuals, alpha=0.3)
    ax.axhline(0)
    ax.set_xlabel("True price")
    ax.set_ylabel("Residual")
    ax.set_title("Residual vs True Price")
    _save_or_close(fig, "vs_true")

    # Residual vs car_age (if exists)
    if "car_age" in X_test.columns:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.scatter(X_test["car_age"], residuals, alpha=0.3)
        ax.axhline(0)
        ax.set_xlabel("Car Age")
        ax.set_ylabel("Residual")
        ax.set_title("Residual vs Car Age")
        _save_or_close(fig, "vs_car_age")

    # Residual vs mileage_per_year (if exists)
    if "mileage_per_year" in X_test.columns:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.scatter(X_test["mileage_per_year"], residuals, alpha=0.3)
        ax.axhline(0)
        ax.set_xlabel("Mileage per Year")
        ax.set_ylabel("Residual")
        ax.set_title("Residual vs Mileage per Year")
        _save_or_close(fig, "vs_mileage_per_year")

    # Residual Distribution
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(residuals, bins=50)
    ax.set_xlabel("Residual")
    ax.set_ylabel("Count")
    ax.set_title("Residual Distribution")
    _save_or_close(fig, "distribution")

def get_feature_importance(model: Pipeline) -> pd.DataFrame:
    """
    Extract feature importance from RandomForest inside pipeline.
    Returns sorted DataFrame of features and their importance.
    """
    # Get preprocessor and regressor
    preprocessor = model.named_steps["preprocessor"]
    regressor = model.named_steps["regressor"]

    if not hasattr(regressor, "feature_importances_"):
        raise ValueError("Regressor does not support feature_importances_")

    # Get feature names after preprocessing
    feature_names = preprocessor.get_feature_names_out()

    importances = regressor.feature_importances_

    df_importance = pd.DataFrame({
        "feature": feature_names,
        "importance": importances
    }).sort_values("importance", ascending=False)

    return df_importance