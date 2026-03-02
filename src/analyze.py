import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline


def analyze_residuals(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    logger
) -> None:
    """
    Residual diagnostics:
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

    # Residual vs Predicted
    plt.figure(figsize=(6, 4))
    plt.scatter(y_pred, residuals, alpha=0.3)
    plt.axhline(0)
    plt.xlabel("Predicted price")
    plt.ylabel("Residual (y_true - y_pred)")
    plt.title("Residual vs Predicted")
    plt.show()

    # Residual vs True
    plt.figure(figsize=(6, 4))
    plt.scatter(y_test, residuals, alpha=0.3)
    plt.axhline(0)
    plt.xlabel("True price")
    plt.ylabel("Residual")
    plt.title("Residual vs True Price")
    plt.show()

    # Residual vs car_age (if exists)
    if "car_age" in X_test.columns:
        plt.figure(figsize=(6, 4))
        plt.scatter(X_test["car_age"], residuals, alpha=0.3)
        plt.axhline(0)
        plt.xlabel("Car Age")
        plt.ylabel("Residual")
        plt.title("Residual vs Car Age")
        plt.show()

    # Residual vs mileage_per_year (if exists)
    if "mileage_per_year" in X_test.columns:
        plt.figure(figsize=(6, 4))
        plt.scatter(X_test["mileage_per_year"], residuals, alpha=0.3)
        plt.axhline(0)
        plt.xlabel("Mileage per Year")
        plt.ylabel("Residual")
        plt.title("Residual vs Mileage per Year")
        plt.show()

    # Residual Distribution
    plt.figure(figsize=(6, 4))
    plt.hist(residuals, bins=50)
    plt.xlabel("Residual")
    plt.ylabel("Count")
    plt.title("Residual Distribution")
    plt.show()

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