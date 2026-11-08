import chess
import numpy as np
import pytest
from pgx.experimental.chess import from_fen

from ws_zero.game.moves import NUM_ACTIONS, action_to_move, move_to_action

# FENs that exercise castling, en passant and every promotion type, for both colors.
SPECIAL_FENS = [
    "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1",  # white can castle both ways
    "r3k2r/8/8/8/8/8/8/R3K2R b KQkq - 0 1",  # black can castle both ways
    "rnbqkbnr/ppp1p1pp/8/3pPp2/8/8/PPPP1PPP/RNBQKBNR w KQkq f6 0 3",  # white en passant
    "rnbqkbnr/pppp1ppp/8/8/3Pp3/8/PPP1PPPP/RNBQKBNR b KQkq d3 0 2",  # black en passant
    "1n1n4/2P5/8/8/8/8/8/K6k w - - 0 1",  # white promotes: push and both captures
    "K6k/8/8/8/8/8/2p5/1N1N4 b - - 0 1",  # black promotes: push and both captures
    "1n6/P7/8/8/8/8/8/K6k w - - 0 1",  # a-file promotion
    "6n1/7P/8/8/8/8/8/K6k w - - 0 1",  # h-file promotion
]


def test_there_are_4672_actions():
    assert NUM_ACTIONS == 4672


def test_initial_position_maps_to_pgx_initial_legal_actions():
    # pgx hard-codes the 20 legal actions of the initial position.
    board = chess.Board()
    ours = sorted(move_to_action(board, m) for m in board.legal_moves)
    assert ours == [89, 90, 652, 656, 673, 674, 1257, 1258, 1841, 1842, 2425, 2426, 3009, 3010, 3572, 3576, 3593,
                    3594, 4177, 4178]


@pytest.mark.parametrize("uci, action", [
    ("e1g1", 2367),  # white king-side castling = king two squares right
    ("e1c1", 2364),  # white queen-side castling
])
def test_white_castling_actions(uci, action):
    board = chess.Board(SPECIAL_FENS[0])
    assert move_to_action(board, chess.Move.from_uci(uci)) == action


@pytest.mark.parametrize("uci, action", [("e8g8", 2367), ("e8c8", 2364)])
def test_black_castling_is_mirrored_to_the_same_actions(uci, action):
    board = chess.Board(SPECIAL_FENS[1])
    assert move_to_action(board, chess.Move.from_uci(uci)) == action


@pytest.mark.parametrize("uci, action", [
    ("c7c8r", 1606),  # underpromotion planes: plane = 3 * piece (R=0, B=1, N=2) + direction (straight=0, right=1, left=2)
    ("c7d8b", 1610),
    ("c7b8n", 1614),
    ("c7c8q", 1622),  # queen promotions use the ordinary queen-move planes
    ("c7d8q", 1650),
    ("c7b8q", 1663),
])
def test_white_promotion_actions(uci, action):
    board = chess.Board(SPECIAL_FENS[4])
    assert move_to_action(board, chess.Move.from_uci(uci)) == action


def test_black_promotion_is_mirrored():
    board = chess.Board(SPECIAL_FENS[5])
    assert move_to_action(board, chess.Move.from_uci("c2c1r")) == 1606


@pytest.mark.parametrize("fen", SPECIAL_FENS)
def test_every_legal_move_round_trips(fen):
    board = chess.Board(fen)
    actions = [move_to_action(board, m) for m in board.legal_moves]
    assert len(set(actions)) == len(actions)
    assert all(0 <= a < NUM_ACTIONS for a in actions)
    for move, action in zip(board.legal_moves, actions):
        assert action_to_move(board, action) == move


@pytest.mark.parametrize("fen", SPECIAL_FENS)
def test_mapped_legal_moves_equal_pgx_legal_action_mask(fen):
    board = chess.Board(fen)
    ours = sorted(move_to_action(board, m) for m in board.legal_moves)
    pgx_mask = np.asarray(from_fen(fen).legal_action_mask)
    assert ours == np.flatnonzero(pgx_mask).tolist()
