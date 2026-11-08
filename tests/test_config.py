import pytest

from ws_zero.config import load_config


def _write(path, text):
    path.write_text(text)
    return path


def test_loads_a_single_yaml_file(tmp_path):
    base = _write(tmp_path / "base.yaml", "seed: 1\nnet:\n  blocks: 6\n  filters: 64\n")
    assert load_config(base) == {"seed": 1, "net": {"blocks": 6, "filters": 64}}


def test_override_replaces_nested_values_and_keeps_siblings(tmp_path):
    base = _write(tmp_path / "base.yaml", "seed: 1\nnet:\n  blocks: 6\n  filters: 64\n")
    arm = _write(tmp_path / "arm.yaml", "net:\n  filters: 32\n")
    assert load_config(base, arm) == {"seed": 1, "net": {"blocks": 6, "filters": 32}}


def test_project_base_config_loads():
    from pathlib import Path

    config = load_config(Path(__file__).parents[1] / "configs" / "base.yaml")
    assert config["init"]["method"] == "random"
    assert config["net"]["batch_norm"] is False


def test_override_cannot_add_keys_missing_from_base(tmp_path):
    # Arms may only change existing settings, so a typo cannot silently create an arm-only setting.
    base = _write(tmp_path / "base.yaml", "net:\n  filters: 64\n")
    arm = _write(tmp_path / "arm.yaml", "net:\n  filtres: 32\n")
    with pytest.raises(KeyError, match="net.filtres"):
        load_config(base, arm)
