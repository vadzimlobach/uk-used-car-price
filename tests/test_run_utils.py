from pathlib import Path
import yaml
import pytest

import src.run_utils as run_utils


class _FixedDateTime:
    """Drop-in replacement for datetime class with a fixed .now()."""

    @classmethod
    def now(cls):
        # 2026-03-05 12:34:56
        from datetime import datetime as _dt

        return _dt(2026, 3, 5, 12, 34, 56)

    @classmethod
    def strftime(cls, *args, **kwargs):
        # Not used directly; run_utils calls datetime.now().strftime(...)
        raise NotImplementedError


def test_create_run_dir_creates_timestamped_dir(tmp_path, monkeypatch):
    # Arrange: make timestamp deterministic
    monkeypatch.setattr(run_utils, "datetime", _FixedDateTime)

    base_dir = tmp_path / "artifacts" / "runs"
    run_name = "rf_baseline"

    # Act
    run_dir = run_utils.create_run_dir(str(base_dir), run_name)

    # Assert
    assert run_dir.exists()
    assert run_dir.is_dir()
    assert run_dir.parent == base_dir
    assert run_dir.name == "20260305_123456_rf_baseline"


def test_create_run_dir_creates_unique_dir_on_collision(tmp_path, monkeypatch):
    """If the same timestamp/run_name is used twice, create_run_dir should not overwrite.
    It should create a different directory (e.g. with __2 suffix) or otherwise remain unique.
    """
    monkeypatch.setattr(run_utils, "datetime", _FixedDateTime)

    base_dir = tmp_path / "artifacts" / "runs"
    run_name = "rf_baseline"

    first = run_utils.create_run_dir(str(base_dir), run_name)
    second = run_utils.create_run_dir(str(base_dir), run_name)

    assert first.exists() and first.is_dir()
    assert second.exists() and second.is_dir()
    assert first != second
    assert first.name == "20260305_123456_rf_baseline"
    assert second.name == "20260305_123456_rf_baseline__2"


def test_save_config_copy_writes_config_yaml(tmp_path):
    run_dir = tmp_path / "some_run"
    run_dir.mkdir(parents=True)

    cfg = {
        "run_name": "rf_baseline",
        "random_state": 42,
        "model": {"rf": {"params": {"n_estimators": 10}}},
        "analysis": {"save_figures": True},
    }

    # Act
    run_utils.save_config_copy(cfg, run_dir)

    # Assert
    p = run_dir / "config.yaml"
    assert p.exists()
    loaded = yaml.safe_load(p.read_text())
    assert loaded == cfg


def test_update_latest_run_writes_pointer_file(tmp_path):
    base_dir = tmp_path / "artifacts" / "runs"
    base_dir.mkdir(parents=True)

    run_id = "20260305_123456_rf_baseline"

    # Act
    run_utils.update_latest_run(str(base_dir), run_id)

    # Assert
    latest = base_dir / "latest_run.txt"
    assert latest.exists()
    assert latest.read_text() == run_id