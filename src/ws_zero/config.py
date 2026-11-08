"""Experiment configs: a base YAML file, optionally overridden by per-arm files that may only change existing keys."""
from pathlib import Path

import yaml


def load_config(base: str | Path, *overrides: str | Path) -> dict:
    """Load `base`, then apply each override in order (nested dicts merge; other values replace)."""
    config = yaml.safe_load(Path(base).read_text()) or {}
    for path in overrides:
        _merge(config, yaml.safe_load(Path(path).read_text()) or {}, prefix="")
    return config


def _merge(into: dict, update: dict, prefix: str) -> None:
    for key, value in update.items():
        dotted = f"{prefix}{key}"
        if key not in into:
            raise KeyError(f"override sets '{dotted}', which is not in the base config")
        if isinstance(into[key], dict) and isinstance(value, dict):
            _merge(into[key], value, prefix=f"{dotted}.")
        else:
            into[key] = value
