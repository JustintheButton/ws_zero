import json
import re

import yaml

from ws_zero.runlog import RunLogger

CONFIG = {"seed": 42, "net": {"blocks": 6, "filters": 64}}


def test_run_folder_contains_the_config(tmp_path):
    with RunLogger(tmp_path, "smoke", CONFIG) as run:
        assert run.run_dir.parent == tmp_path
        assert yaml.safe_load((run.run_dir / "config.yaml").read_text()) == CONFIG


def test_meta_records_provenance(tmp_path):
    with RunLogger(tmp_path, "smoke", CONFIG) as run:
        meta = json.loads((run.run_dir / "meta.json").read_text())
    assert meta["seed"] == 42
    assert meta["name"] == "smoke"
    for key in ("git_commit", "git_dirty", "started_at", "host", "python", "jax", "jaxlib", "pgx", "numpy", "devices"):
        assert key in meta
    assert "torch" not in meta
    assert meta["git_commit"] is None or re.fullmatch(r"[0-9a-f]{40}", meta["git_commit"])


def test_metrics_are_appended_as_json_lines(tmp_path):
    with RunLogger(tmp_path, "smoke", CONFIG) as run:
        run.log(0, loss=2.5)
        run.log(10, loss=1.25, games=3)
        lines = (run.run_dir / "metrics.jsonl").read_text().splitlines()
    assert [json.loads(line) for line in lines] == [
        {"step": 0, "loss": 2.5},
        {"step": 10, "loss": 1.25, "games": 3},
    ]


def test_runs_with_the_same_name_get_separate_folders(tmp_path):
    with RunLogger(tmp_path, "smoke", CONFIG) as a, RunLogger(tmp_path, "smoke", CONFIG) as b:
        assert a.run_dir != b.run_dir


def test_wandb_mirror_runs_offline_into_the_run_folder(tmp_path, monkeypatch):
    monkeypatch.setenv("WANDB_SILENT", "true")
    with RunLogger(tmp_path, "smoke", CONFIG, wandb_mode="offline") as run:
        run.log(0, loss=2.5)
    assert (run.run_dir / "wandb").is_dir()
