"""The arena's browser code (arena/js/) against Python: the observation, legal moves and outcomes, the network in
TensorFlow.js, and the search. Runs arena/tests/check.js in Node; skipped if Node or the downloaded libraries
(build/vendor, fetched by scripts/build_arena.py) are missing."""
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from ws_zero.arena import ENCODING_LINES, encoding_cases, network_cases, network_lines, pawn_pusher, untrained
from ws_zero.export import write_model
from ws_zero.game.moves import NUM_ACTIONS

ROOT = Path(__file__).resolve().parents[1]
VENDOR = [ROOT / "build/vendor/chess.min.js", ROOT / "build/vendor/tf.min.js"]

SEARCH_LINES = [  # (FEN, moves) searched with the material counter
    ("6k1/5ppp/8/8/8/8/5PPP/R5K1 w - - 0 1", []),  # mate in one
    ("r1bqkbnr/pppp1ppp/8/4n3/4P3/5Q2/PPPP1PPP/RNB1KBNR w KQkq - 0 4", []),  # the gate's knight position
    ("8/5k2/8/3p4/3P4/2K5/8/8 w - - 0 1", []),  # pawn ending
    (None, ["e2e4", "e7e5", "g1f3"]),  # Black to move, with history
    ENCODING_LINES[3][0:1] + (ENCODING_LINES[3][1][:6],),  # one move from threefold repetition
]


def _material(obs):
    values = np.array([1, 3, 3, 5, 9, 0], dtype=np.float64)
    diff = (obs[..., 0:6].astype(np.float64) @ values).sum() - (obs[..., 6:12].astype(np.float64) @ values).sum()
    return np.zeros(NUM_ACTIONS, np.float64), float(np.tanh(diff / 4))


def _search_case(fen, moves, simulations):
    from reference_search import reference_search
    root = reference_search(fen, _material, simulations, moves)
    return {"fen": fen, "moves": list(moves), "simulations": simulations,
            "visits": {str(a): c.visits for a, c in sorted(root.children.items()) if c.visits}}


@pytest.mark.skipif(shutil.which("node") is None or not all(p.exists() for p in VENDOR),
                    reason="needs Node and build/vendor (run scripts/build_arena.py once)")
def test_the_browser_code_matches_python(tmp_path):
    from test_export import _trained_looking_params
    networks = []
    for name, params in [("untrained", untrained()), ("pawn_pusher", pawn_pusher()),
                         ("trained_looking", _trained_looking_params())]:
        write_model(tmp_path / name, params, {"name": name})
        networks.append({"model": str(tmp_path / name), "cases": network_cases(params, network_lines())})
    searches = [_search_case(f, m, 100) for f, m in SEARCH_LINES]
    cases = {"encoding": encoding_cases(), "networks": networks, "searches": searches}
    (tmp_path / "cases.json").write_text(json.dumps(cases))
    run = subprocess.run(["node", str(ROOT / "arena/tests/check.js"), str(tmp_path / "cases.json")],
                         capture_output=True, text=True, timeout=600)
    print(run.stdout)
    assert run.returncode == 0, run.stdout + run.stderr
    assert "all checks passed" in run.stdout
