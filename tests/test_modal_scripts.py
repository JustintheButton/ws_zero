"""The Modal launch scripts: a GPU name the price table does not know must stop the script before any paid run."""
import importlib.util
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def _load(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _NoPaidRun:
    def with_options(self, **_):
        raise AssertionError("a paid run was started")


@pytest.mark.parametrize("script, function, argument", [
    ("modal_selfplay_speed", "measure", {"gpu": "A10G"}),  # Modal's name; the price table calls it A10
    ("modal_bench", "bench", {"gpus": "L4,A10G"}),
])
def test_an_unknown_gpu_name_stops_the_script_before_any_paid_run(script, function, argument, monkeypatch):
    pytest.importorskip("modal")
    module = _load(script)
    monkeypatch.setattr(module, function, _NoPaidRun())
    with pytest.raises(SystemExit, match="A10G"):
        module.main.info.raw_f(**argument)
