import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import src.train as train_mod


class DummyLogger:
    def info(self, *args, **kwargs):
        pass

    def warning(self, *args, **kwargs):
        pass


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


def test_set_X_y_splits_correctly():
    df = _toy_df()
    train_df = df.iloc[:24].copy()
    test_df = df.iloc[24:].copy()

    X_train, y_train, X_test, y_test = train_mod.set_X_y(train_df, test_df, target="price")

    assert "price" not in X_train.columns
    assert "price" not in X_test.columns
    assert y_train.name == "price"
    assert y_test.name == "price"
    assert len(X_train) == 24
    assert len(X_test) == 6


def test_cross_validate_model_returns_expected_keys_fast():
    df = _toy_df()
    X = df.drop(columns=["price"])
    y = df["price"]

    # small/faster model config through pipeline builder (use your own build_model_pipeline)
    config = {
        "model_type": "rf",
        "random_state": 42,
        "model": {"rf": {"params": {"n_estimators": 5, "max_depth": 3}}},
    }
    base_pipeline = train_mod.build_model_pipeline(X, config)

    summary = train_mod.cross_validate_model(
        X,
        y,
        base_pipeline=base_pipeline,
        n_splits=3,
        random_state=42,
        log_target=False,
        shuffle=True,
    )

    for k in ["rmse_mean", "rmse_std", "mae_mean", "mae_std", "r2_mean", "r2_std", "n_splits", "log_target"]:
        assert k in summary

    assert summary["rmse_mean"] >= 0
    assert summary["mae_mean"] >= 0
    assert summary["n_splits"] == 3
    assert summary["log_target"] is False


def test_save_artifacts_writes_metrics_and_model(tmp_path, monkeypatch):
    # Avoid actually serializing with joblib; just assert it was called with correct path
    calls = {"dump": 0}

    def fake_dump(model, out_path):
        calls["dump"] += 1
        # create a placeholder file to mimic dump side-effect
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        Path(out_path).write_bytes(b"dummy")

    monkeypatch.setattr(train_mod.joblib, "dump", fake_dump)

    logger = DummyLogger()
    model_out = tmp_path / "models" / "m.joblib"
    metrics_out = tmp_path / "reports" / "metrics.json"

    train_mod.save_artifacts(
        model="fake_model_obj",
        metrics={"a": 1},
        model_out=model_out,
        metrics_out=metrics_out,
        logger=logger,
    )

    assert calls["dump"] == 1
    assert model_out.exists()
    assert metrics_out.exists()

    payload = json.loads(metrics_out.read_text(encoding="utf-8"))
    assert payload == {"a": 1}


def test_main_smoke(monkeypatch, tmp_path):
    """
    Runs main() end-to-end with everything mocked:
    - config loaded from temp yaml
    - data read mocked to toy df
    - logging mocked
    - residual analysis mocked
    - artifacts saved into tmp_path
    """
    # 1) create temp config yaml
    cfg_path = tmp_path / "train.yaml"
    cfg = {
        "log_level": "INFO",
        "run_name": "test_run",
        "random_state": 42,
        "data": {"input_path": "ignored.csv", "target": "price"},
        "cv": {"n_splits": 3, "shuffle": True},
        "model_type": "rf",
        "model": {"log_target": False, "rf": {"params": {"n_estimators": 5, "max_depth": 3}}},
        "output": {
            "model_path": str(tmp_path / "artifacts" / "models" / "m.joblib"),
            "metrics_path": str(tmp_path / "artifacts" / "reports" / "metrics.json"),
            "cv_summary_path": str(tmp_path / "artifacts" / "reports" / "cv.json"),
        },
    }
    import yaml
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")

    # 2) mock read_data_from_file to return toy df
    monkeypatch.setattr(train_mod, "read_data_from_file", lambda in_path, logger: _toy_df())

    # 3) mock logging + residuals
    monkeypatch.setattr(train_mod, "setup_logging", lambda level: DummyLogger())
    monkeypatch.setattr(train_mod, "analyze_residuals", lambda *args, **kwargs: None)

    # 4) mock argparse to pass our config path
    class Args:
        config = cfg_path

    monkeypatch.setattr(train_mod.argparse.ArgumentParser, "parse_args", lambda self: Args())

    # 5) run main
    train_mod.main()

    # 6) assert artifacts written
    assert Path(cfg["output"]["model_path"]).exists()
    assert Path(cfg["output"]["metrics_path"]).exists()

    metrics_payload = json.loads(Path(cfg["output"]["metrics_path"]).read_text(encoding="utf-8"))
    assert "metrics" in metrics_payload
    assert "config" in metrics_payload