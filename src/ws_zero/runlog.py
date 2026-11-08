"""Per-run logging: a local run folder (the source of truth), optionally mirrored to Weights & Biases."""
import json
import platform
import socket
import subprocess
import sys
import time
from importlib.metadata import version
from pathlib import Path

import jax
import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]


class RunLogger:
    """Creates `root/<timestamp>_<name>/` with config.yaml, meta.json and metrics.jsonl.

    wandb_mode: "disabled" (local only), "offline" (W&B files kept locally) or "online".
    """

    def __init__(self, root, name: str, config: dict, *, wandb_mode: str = "disabled",
                 wandb_project: str = "ws_zero"):
        self.run_dir = _unique_dir(Path(root), f"{time.strftime('%Y%m%d-%H%M%S')}_{name}")
        (self.run_dir / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
        (self.run_dir / "meta.json").write_text(json.dumps(_meta(name, config), indent=2))
        self._metrics = open(self.run_dir / "metrics.jsonl", "a")
        self._wandb = None
        if wandb_mode != "disabled":
            import wandb  # imported lazily: slow to import, and unused in local-only runs
            self._wandb = wandb.init(project=wandb_project, name=self.run_dir.name, config=config,
                                     dir=str(self.run_dir), mode=wandb_mode)

    def log(self, step: int, **metrics) -> None:
        self._metrics.write(json.dumps({"step": step, **metrics}) + "\n")
        self._metrics.flush()
        if self._wandb is not None:
            self._wandb.log(metrics, step=step)

    def close(self) -> None:
        self._metrics.close()
        if self._wandb is not None:
            self._wandb.finish()
            self._wandb = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def _unique_dir(root: Path, stem: str) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    for i in range(1000):
        path = root / (stem if i == 0 else f"{stem}-{i}")
        try:
            path.mkdir()
            return path
        except FileExistsError:
            continue
    raise RuntimeError(f"could not create a unique run folder for {stem}")


def _git(*args) -> str | None:
    try:
        out = subprocess.run(["git", "-C", str(REPO_ROOT), *args], capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return out.stdout.strip() if out.returncode == 0 else None


def _meta(name: str, config: dict) -> dict:
    commit = _git("rev-parse", "HEAD")
    status = _git("status", "--porcelain")
    return {
        "name": name,
        "seed": config.get("seed"),
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "git_commit": commit,
        "git_dirty": None if status is None else bool(status),
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "jax": jax.__version__,
        "jaxlib": version("jaxlib"),
        "pgx": version("pgx"),
        "numpy": np.__version__,
        "devices": [str(d) for d in jax.devices()],
    }
