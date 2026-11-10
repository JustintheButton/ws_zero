"""The opening suites for evaluation games (protocol §8.3), made once by scripts/make_openings.py and committed.

`load()` returns {"checkpoint": [50 openings], "go_no_go": [100 openings], "generator": {...}}. An opening is
{"moves": 8 UCI moves from the start position, "fen": the position reached, "centipawns": Stockfish's evaluation of
it at depth 20, from White's side}. Every evaluation pairing plays each opening twice, colours reversed.
"""
import json
from pathlib import Path

PATH = Path(__file__).resolve().parents[2] / "data" / "openings.json"


def load(path: Path | None = None) -> dict:
    return json.loads(Path(path or PATH).read_text())
