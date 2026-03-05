import yaml
import subprocess
from datetime import datetime
from pathlib import Path

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
    with open(config_path, 'w') as f:
        yaml.safe_dump(config, f)

def update_latest_run(base_dir: str, run_id: str) -> None:
    base = Path(base_dir)
    base.mkdir(parents=True, exist_ok=True)  # ✅ ensure directory exists
    latest_file = base / "latest_run.txt"
    latest_file.write_text(run_id)


def add_link_to_code_version(run_dir:Path) -> None:
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
    (run_dir / "git_commit.txt").write_text(commit)