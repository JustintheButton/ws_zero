"""The opening suites (protocol §8.3): random 8-ply lines that Stockfish at depth 20 rates within half a pawn."""
import importlib.util
from pathlib import Path

import chess
import pytest

from ws_zero import openings, stockfish

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "make_openings.py"


def _script():
    spec = importlib.util.spec_from_file_location("make_openings", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _final(moves):
    board = chess.Board()
    for uci in moves:
        board.push_uci(uci)  # raises if a move is illegal
    return board


def test_candidate_lines_are_eight_legal_plies_unfinished_never_the_same_position_twice_and_repeat_by_seed():
    make = _script()
    lines = [line for line, _ in zip(make.candidate_lines(seed=5), range(300))]
    finals = [_final(line) for line in lines]
    assert all(len(line) == 8 for line in lines) and not any(b.is_game_over() for b in finals)
    assert len({" ".join(b.fen().split()[:4]) for b in finals}) == 300
    assert lines == [line for line, _ in zip(make.candidate_lines(seed=5), range(300))]
    assert lines != [line for line, _ in zip(make.candidate_lines(seed=6), range(300))]


def test_selection_keeps_the_first_lines_within_the_limit_in_order():
    make = _script()
    lines = [[f"line{i}"] for i in range(12)]
    scores = [0, 51, -50, 300, 49, -51, 50, 10, None, -20, 5, 7]  # None: a forced mate
    kept = make.select(zip(lines, scores), count=5, limit=50)
    assert kept == [(["line0"], 0), (["line2"], -50), (["line4"], 49), (["line6"], 50), (["line7"], 10)]


@pytest.mark.skipif(stockfish.installed_engine() is None, reason="Stockfish is not installed")
def test_stockfish_rates_positions_from_whites_side_in_centipawns():
    with stockfish.Analyst(depth=8) as analyst:
        assert abs(analyst.centipawns(chess.Board())) <= 80
        assert analyst.centipawns(chess.Board("rnb1kbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")) > 500
        assert analyst.centipawns(chess.Board("rnb1kbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR b KQkq - 0 1")) > 500
        assert analyst.centipawns(chess.Board("6k1/5ppp/8/8/8/8/5PPP/R5K1 w - - 0 1")) is None  # mate in one


def test_the_committed_suites_are_50_and_100_different_balanced_openings():
    suites = openings.load()
    checkpoint, go_no_go = suites["checkpoint"], suites["go_no_go"]
    assert (len(checkpoint), len(go_no_go)) == (50, 100)
    positions = set()
    for opening in checkpoint + go_no_go:
        board = _final(opening["moves"])
        assert len(opening["moves"]) == 8 and not board.is_game_over()
        assert board.fen() == opening["fen"] and abs(opening["centipawns"]) <= 50
        positions.add(" ".join(board.fen().split()[:4]))
    assert len(positions) == 150  # no position twice, within or across the suites
