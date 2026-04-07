import json
import shutil
from pathlib import Path

import yaml


def main() -> None:
    artifacts_dir = Path("artifacts")
    runs_dir = artifacts_dir / "runs"
    serving_dir = artifacts_dir / "serving"
    serving_dir.mkdir(parents=True, exist_ok=True)

    latest_run_file = runs_dir / "latest_run.txt"
    run_id = latest_run_file.read_text(encoding="utf-8").strip()
    run_dir = runs_dir / run_id

    model_path = run_dir / "model.joblib"
    git_commit_path = run_dir / "git_commit.txt"
    config_path = run_dir / "config.yaml"

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    if not git_commit_path.exists():
        raise FileNotFoundError(f"Git commit file not found: {git_commit_path}")
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    shutil.copy2(model_path, serving_dir / "model.joblib")

    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    metadata = {
        "run_id": run_id,
        "git_commit": git_commit_path.read_text(encoding="utf-8").strip(),
        "model_type": config.get("model_type"),
    }

    (serving_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    print(f"Prepared serving artifacts for run: {run_id}")


if __name__ == "__main__":
    main()
