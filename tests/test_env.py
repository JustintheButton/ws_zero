import chess
import jax
import numpy as np
import pytest

from ws_zero.game.env import ChessEnv, state_from_fen, state_to_fen
from ws_zero.game.moves import NUM_ACTIONS, move_to_action
from ws_zero.game.reference import ReferenceGame, observations_match

ENV = ChessEnv()
FOOLS_MATE = ["f2f3", "e7e5", "g2g4", "d8h4"]


def _play(ucis, batch=1):
    """Play the same moves in every game of a batch, and in the reference."""
    state, ref = ENV.init(batch), ReferenceGame()
    for uci in ucis:
        move = chess.Move.from_uci(uci)
        state = ENV.step(state, np.full(batch, move_to_action(ref.board, move)))
        ref.push(move)
    return state, ref


def test_initial_batch_matches_the_reference():
    state, ref = _play([], batch=3)
    assert ENV.legal_action_mask(state).shape == (3, NUM_ACTIONS)
    for i in range(3):
        assert np.flatnonzero(ENV.legal_action_mask(state)[i]).tolist() == ref.legal_actions()
        np.testing.assert_array_equal(ENV.observe(state)[i], ref.observation())
    assert not ENV.is_terminal(state).any()


def test_repetition_of_the_start_position_is_detected():
    # Regression: under JAX >= 0.5, pgx's hard-coded start-position hash no longer matches its own hash function.
    shuffle = ["g1f3", "g8f6", "f3g1", "f6g8"]
    state, ref = _play(shuffle)
    np.testing.assert_array_equal(ENV.observe(state)[0], ref.observation())
    state, _ = _play(shuffle * 2)  # start position for the third time
    assert ENV.is_terminal(state)[0]


def test_checkmate_gives_minus_one_to_the_side_to_move():
    state, _ = _play(FOOLS_MATE)
    assert ENV.is_terminal(state)[0]
    assert ENV.terminal_value(state)[0] == -1.0


def test_finished_games_are_not_changed_by_further_steps():
    state, _ = _play(FOOLS_MATE)
    after = ENV.step(state, np.array([0]))
    np.testing.assert_array_equal(ENV.observe(after), ENV.observe(state))
    assert ENV.is_terminal(after)[0] and ENV.terminal_value(after)[0] == -1.0


@pytest.mark.parametrize("fen", [
    "r3k2r/8/8/8/8/8/8/R3K2R b KQkq - 0 1",
    "rnbqkbnr/pppp1ppp/8/8/3Pp3/8/PPP1PPPP/RNBQKBNR b KQkq d3 0 2",
    "1n1n4/2P5/8/8/8/8/8/K6k w - - 7 40",
])
def test_fen_round_trip(fen):
    assert state_to_fen(state_from_fen(fen)) == fen


@pytest.mark.parametrize("fen, ucis", [
    ("r3k2r/pppq1ppp/2npbn2/2b1p3/2B1P3/2NPBN2/PPPQ1PPP/R3K2R w KQkq - 0 8", ["e1g1", "e8c8"]),
    ("r1n1k3/1P6/8/8/8/8/6p1/4K2R b K - 3 1", ["g2h1q", "e1d2"]),
])
def test_game_from_fen_has_no_history_before_its_start(fen, ucis):
    # Regression: pgx's from_fen keeps the default history, so the standard start position appeared one ply back.
    state, ref = jax.tree.map(lambda x: x[None], state_from_fen(fen)), ReferenceGame(fen, repetition_key="pgx")
    for uci in [None] + ucis:
        if uci:
            move = chess.Move.from_uci(uci)
            state = ENV.step(state, np.array([move_to_action(ref.board, move)]))
            ref.push(move)
        assert observations_match(np.asarray(ENV.observe(state))[0], ref.observation())
