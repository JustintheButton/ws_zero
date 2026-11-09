"""scripts/install_stockfish.py: a download is kept only if it is the pinned file, and the bench number is read."""
import hashlib
import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "install_stockfish.py"


def _load():
    spec = importlib.util.spec_from_file_location("install_stockfish", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_a_download_with_the_wrong_checksum_is_refused_and_not_kept(tmp_path):
    source = tmp_path / "release.tar.gz"
    source.write_bytes(b"not the file that was pinned")
    wrong = hashlib.sha256(b"the file that was pinned").hexdigest()
    with pytest.raises(SystemExit, match="checksum"):
        _load().download(source.as_uri(), wrong, tmp_path / "kept.tar.gz")
    assert not (tmp_path / "kept.tar.gz").exists()


def test_a_download_with_the_pinned_checksum_is_kept(tmp_path):
    source = tmp_path / "release.tar.gz"
    source.write_bytes(b"the file that was pinned")
    _load().download(source.as_uri(), hashlib.sha256(b"the file that was pinned").hexdigest(), tmp_path / "kept.tar.gz")
    assert (tmp_path / "kept.tar.gz").read_bytes() == b"the file that was pinned"


def test_the_bench_number_is_read_from_stockfish_output():
    output = "Total time (ms) : 1234\nNodes searched  : 2336177\nNodes/second    : 1893174\n"
    assert _load().parse_bench(output) == 2336177
    with pytest.raises(SystemExit, match="bench"):
        _load().parse_bench("Stockfish 19 by the Stockfish developers\n")
