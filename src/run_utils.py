import os
import subprocess
from datetime import datetime
from pathlib import Path

import yaml


def create_run_dir(base_dir: str, run_name: str) -> Path:
    """
    Create a versioned run directory.

    Primary format:
      YYYYMMDD_HHMMSS_<run_name>

    If that already exists (possible in CI), append:
      __2, __3, ...
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = Path(base_dir)
    base.mkdir(parents=True, exist_ok=True)

    stem = f"{timestamp}_{run_name}"

    for i in range(1, 1000):
        suffix = "" if i == 1 else f"__{i}"
        run_id = f"{stem}{suffix}"
        run_dir = base / run_id
        try:
            run_dir.mkdir(parents=True, exist_ok=False)
            return run_dir
        except FileExistsError:
            continue

    raise RuntimeError(f"Could not create a unique run dir for stem: {stem}")


def save_config_copy(config: dict, run_dir: Path) -> None:
    config_path = run_dir / "config.yaml"
    with open(config_path, "w") as f:
        yaml.safe_dump(config, f)


def update_latest_run(base_dir: str, run_id: str) -> None:
    base = Path(base_dir)
    base.mkdir(parents=True, exist_ok=True)  # ✅ ensure directory exists
    latest_file = base / "latest_run.txt"
    latest_file.write_text(run_id)


def add_link_to_code_version(run_dir: Path) -> None:
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
    (run_dir / "git_commit.txt").write_text(commit)


def resolve_latest_model_path() -> Path:
    latest_file = Path("artifacts/runs/latest_run.txt")
    if not latest_file.exists():
        raise SystemExit("latest_run.txt not found. Train model first or set MODEL_PATH.")

    run_id = latest_file.read_text(encoding="utf-8").strip()
    model_path = Path("artifacts/runs") / run_id / "model.joblib"

    if not model_path.exists():
        raise SystemExit(f"Model not found at {model_path}")

    return model_path


def resolve_run_id(model_path: Path) -> str:
    return model_path.parent.name


def read_git_commit_from_run(model_path: Path) -> str:
    commit_path = model_path.parent / "git_commit.txt"
    if commit_path.exists():
        return commit_path.read_text(encoding="utf-8").strip()
    return os.getenv("GIT_COMMIT", "unknown")


def read_model_type_from_run(model_path: Path) -> str | None:
    config_path = model_path.parent / "config.yaml"
    if not config_path.exists():
        return None

    try:
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        return config.get("model_type")
    except Exception:
        return None


def build_model_version(model_path: Path) -> dict:
    return {
        "run_id": resolve_run_id(model_path),
        "git_commit": read_git_commit_from_run(model_path),
        "model_type": read_model_type_from_run(model_path),
    }
