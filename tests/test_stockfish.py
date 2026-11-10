"""ws_zero.stockfish: the pinned Stockfish as an opponent. Skipped where the engine is not installed
(scripts/install_stockfish.py)."""
import stat

import chess
import pytest

from ws_zero import stockfish
from ws_zero.stockfish import Stockfish

needs_engine = pytest.mark.skipif(stockfish.installed_engine() is None,
                                  reason="Stockfish is not installed (scripts/install_stockfish.py)")

POSITIONS = stockfish.PINNED_POSITIONS


@needs_engine
@pytest.mark.parametrize("nodes", [1, 10, 100, 1000, 10000])
def test_the_same_position_and_node_count_always_give_the_same_move(nodes):
    def moves():
        with Stockfish(nodes) as engine:
            return [engine.move(chess.Board(fen)).uci() for fen in POSITIONS]
    first = moves()
    assert moves() == first  # a fresh engine process
    with Stockfish(nodes) as engine:  # one engine asked about each position alone, twice over, in another order
        alone = {fen: engine.move(chess.Board(fen)).uci() for fen in POSITIONS[::-1] + POSITIONS}
    assert [alone[fen] for fen in POSITIONS] == first  # what was searched before does not matter


@needs_engine
def test_a_whole_game_replays_move_for_move():
    def game():
        board = chess.Board()
        with Stockfish(100) as engine:
            while not board.is_game_over() and len(board.move_stack) < 40:
                board.push(engine.move(board))
        return [m.uci() for m in board.move_stack]
    first = game()
    assert len(first) == 40 and game() == first
    board = chess.Board()
    with Stockfish(100) as engine:  # the same game after the engine has played something else
        for fen in POSITIONS:
            engine.move(chess.Board(fen))
        while len(board.move_stack) < 40:
            board.push(engine.move(board))
    assert [m.uci() for m in board.move_stack] == first


@needs_engine
def test_every_move_is_legal_and_a_mate_in_one_is_played_at_1000_nodes():
    board = chess.Board(POSITIONS[2])
    with Stockfish(1000) as engine:
        move = engine.move(board)
    assert move in board.legal_moves
    board.push(move)
    assert board.is_checkmate()


@needs_engine
def test_the_strength_limited_engine_plays_legal_moves():
    board = chess.Board(POSITIONS[1])
    with Stockfish(1000, elo=stockfish.MIN_ELO) as engine:
        for _ in range(6):
            move = engine.move(board)
            assert move in board.legal_moves
            board.push(move)


def test_an_engine_that_is_not_the_pinned_stockfish_is_refused(tmp_path):
    fake = tmp_path / "stockfish"
    fake.write_text('#!/bin/sh\nwhile read line; do case "$line" in\n'
                    '  uci) echo "id name Stockfish 8"; echo "uciok";;\n  isready) echo "readyok";;\n  quit) exit 0;;\n'
                    'esac; done\n')
    fake.chmod(fake.stat().st_mode | stat.S_IXUSR)
    with pytest.raises(RuntimeError, match="Stockfish 19"):
        Stockfish(100, path=fake)


@needs_engine
def test_this_machine_plays_the_pinned_moves():
    stockfish.check_pinned_moves()


def test_an_engine_that_plays_other_moves_fails_the_pinned_check(monkeypatch):
    if stockfish.installed_engine() is None:
        pytest.skip("Stockfish is not installed")
    monkeypatch.setitem(stockfish.PINNED_MOVES, 1000, ["d2d4", "g8f6", "h5f7", "c3c2"])  # as if pinned elsewhere
    with pytest.raises(RuntimeError, match="1000 nodes"):
        stockfish.check_pinned_moves()
