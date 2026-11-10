"""scripts/ladder_matches.py: one rating per player from all the pairings together."""
import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "ladder_matches.py"


def _load():
    spec = importlib.util.spec_from_file_location("ladder_matches", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_ratings_put_a_75_percent_score_191_elo_apart_and_chain_through_a_middle_player():
    # b scores 75% against a, c scores 75% against b; a and c never meet. 75% is 400 x log10(3) = 190.8 Elo.
    pairings = [{"low": "a", "high": "b", "games": 200, "score": 0.75},
                {"low": "b", "high": "c", "games": 200, "score": 0.75}]
    ratings = _load().fit_ratings(["a", "b", "c"], pairings)
    assert ratings["a"] == 0
    assert ratings["b"] == pytest.approx(190.8, abs=0.5)
    assert ratings["c"] == pytest.approx(381.7, abs=1.0)
