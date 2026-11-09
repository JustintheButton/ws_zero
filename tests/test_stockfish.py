"""ws_zero.stockfish: the pinned Stockfish as an opponent. Skipped where the engine is not installed
(scripts/install_stockfish.py)."""
import stat

import chess
import pytest

from ws_zero import stockfish
from ws_zero.stockfish import Stockfish

needs_engine = pytest.mark.skipif(stockfish.installed_engine() is None,
                                  reason="Stockfish is not installed (scripts/install_stockfish.py)")

POSITIONS = [
    chess.STARTING_FEN,
    "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3",  # Italian
    "r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4",  # Qxf7# is there
    "8/5k2/8/3p4/3P4/2K5/8/8 w - - 0 1",  # pawn ending
]


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


# What the Mac's engine played on 2026-11-09. Stockfish at one thread and a fixed node count is the same on every
# machine, so Modal and any other machine must reproduce these exactly; a difference means a different engine.
PINNED_MOVES = {  # nodes -> the move in each of POSITIONS
    1: ["d2d4", "g8f6", "h5f7", "c3b4"],
    10: ["d2d4", "g8f6", "h5f7", "c3b4"],
    100: ["d2d4", "g8f6", "h5f7", "c3d2"],
    1000: ["e2e4", "g8f6", "h5f7", "c3c2"],
    10000: ["e2e4", "g8f6", "h5f7", "c3b4"],
}
PINNED_GAME = (  # Stockfish against itself at 100 nodes, 40 plies from the start
    "d2d4 d7d5 g1f3 g8f6 c2c3 c8f5 d1b3 d8c8 c1f4 b8d7 b1d2 a7a5 e2e3 e7e6 a2a4 f6h5 f4e5 f8e7 f1b5 c7c6 "
    "b5e2 h5f6 e5f4 h7h6 h2h3 e8g8 f3h4 f5h7 h4f3 g8h8 e1g1 f8g8 f3e5 d7e5 f4e5 f6d7 e5g3 g8f8 c3c4 e7b4")


@needs_engine
def test_this_machine_plays_the_pinned_moves():
    for nodes, expected in PINNED_MOVES.items():
        with Stockfish(nodes) as engine:
            assert [engine.move(chess.Board(fen)).uci() for fen in POSITIONS] == expected, nodes
    board = chess.Board()
    with Stockfish(100) as engine:
        while len(board.move_stack) < 40:
            board.push(engine.move(board))
    assert " ".join(m.uci() for m in board.move_stack) == PINNED_GAME
