import yaml
import subprocess
from datetime import datetime
from pathlib import Path

def create_run_dir(base_dir: str, run_name: str) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"{timestamp}_{run_name}"

    run_dir = Path(base_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir

def save_config_copy(config: dict, run_dir: Path) -> None:
    config_path = run_dir / "config.yaml"
    with open(config_path, 'w') as f:
        yaml.safe_dump(config, f)

def update_latest_run(base_dir:str, run_id: str) -> None:
    latest_path = Path(base_dir) / "latest_run.txt"
    latest_path.write_text(run_id)


def add_link_to_code_version(run_dir:Path) -> None:
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
    (run_dir / "git_commit.txt").write_text(commit)