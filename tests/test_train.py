import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

import src.train as train_mod
from src.train import main


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


def test_main_smoke_creates_versioned_run(tmp_path, monkeypatch):
    # --- create tiny dataset on the fly ---
    df = pd.DataFrame(
        {
            # target
            "price": [5000, 6000, 7000, 8000, 9000, 10000],
            # minimal numeric features (should work with RF)
            "car_age": [10, 9, 8, 7, 6, 5],
            "mileage_per_year": [12000, 11000, 10000, 9000, 8000, 7000],
        }
    )
    data_path = tmp_path / "processed_small.csv"
    df.to_csv(data_path, index=False)

    # --- config pointing to tmp data + tmp artifacts dir ---
    config = {
        "run_name": "rf_baseline",
        "log_level": "INFO",
        "random_state": 42,
        "model_type": "rf",
        "data": {"input_path": str(data_path), "target": "price"},
        "model": {"rf": {"params": {"n_estimators": 5, "max_depth": 3}}},
        "cv": {"n_splits": 2, "shuffle": True},
        "analysis": {"save_figures": False},
        "artifacts_dir": str(tmp_path / "artifacts" / "runs"),
    }

    cfg_path = tmp_path / "train.yaml"
    cfg_path.write_text(yaml.safe_dump(config))

    # run main as CLI
    monkeypatch.setattr("sys.argv", ["train", "--config", str(cfg_path)])
    main()
    runs_base = Path(config["artifacts_dir"])
    assert runs_base.exists()

    # assertions: latest pointer and expected files in run dir
    runs_base = Path(config["artifacts_dir"])
    latest = runs_base / "latest_run.txt"
    assert latest.exists()

    run_id = latest.read_text().strip()
    run_dir = runs_base / run_id
    assert run_dir.exists() and run_dir.is_dir()

    assert (run_dir / "model.joblib").exists()
    assert (run_dir / "metrics.json").exists()
    assert (run_dir / "cv_summary.json").exists()
    assert (run_dir / "config.yaml").exists()