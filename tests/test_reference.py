import chess
import jax
import numpy as np
import pytest
from pgx.experimental.chess import Game

from ws_zero.game.env import initial_state
from ws_zero.game.moves import move_to_action
from ws_zero.game.reference import ReferenceGame, observations_match

GAME = Game()
STEP = jax.jit(GAME.step)
OBSERVE = jax.jit(GAME.observe)


def _play_both(ucis):
    """Play the same moves in the reference and in pgx (with our start-position hash fix); return both."""
    ref, state = ReferenceGame(), initial_state()
    for uci in ucis:
        move = chess.Move.from_uci(uci)
        state = STEP(state, move_to_action(ref.board, move))
        ref.push(move)
    return ref, state


def test_initial_observation_matches_pgx():
    ref, state = _play_both([])
    np.testing.assert_array_equal(ref.observation(), np.asarray(OBSERVE(state, state.color)))


def test_observation_matches_pgx_with_full_history_and_a_repetition():
    # Knight shuffle repeats the start position, then more moves fill all 8 history slots.
    ucis = ["g1f3", "g8f6", "f3g1", "f6g8", "e2e4", "e7e5", "g1f3", "b8c6", "f1b5", "a7a6", "e1g1"]
    ref, state = _play_both(ucis)
    np.testing.assert_array_equal(ref.observation(), np.asarray(OBSERVE(state, state.color)))


REPEAT_AFTER_DOUBLE_PUSH = ["f2f4", "g8f6", "g1f3", "f6g8", "f3g1"]  # repeats the position right after f4


def test_pgx_repetition_key_counts_a_bare_en_passant_square():
    # pgx's hash includes the en passant square after every double pawn push, even when no capture is possible.
    ref, state = ReferenceGame(repetition_key="pgx"), initial_state()
    for uci in REPEAT_AFTER_DOUBLE_PUSH:
        move = chess.Move.from_uci(uci)
        state = STEP(state, move_to_action(ref.board, move))
        ref.push(move)
    np.testing.assert_array_equal(ref.observation(), np.asarray(OBSERVE(state, state.color)))


def test_fide_repetition_key_ignores_an_impossible_en_passant():
    ref = ReferenceGame()  # default: FIDE
    for uci in REPEAT_AFTER_DOUBLE_PUSH:
        ref.push(chess.Move.from_uci(uci))
    obs = ref.observation()
    assert obs[0, 0, 13] == 1.0  # current position: repeated once


def test_observations_match_tolerates_rounding_only_in_the_two_scalar_planes():
    # XLA may compute x / 100 as x * 0.01, so planes 113 (plies / 512) and 118 (no-progress / 100) can differ by an ulp.
    a = ReferenceGame("8/8/4k3/8/8/4K3/4R3/8 w - - 5 80").observation()
    b = a.copy()
    b[..., 118] = np.float32(5) * np.float32(0.01)
    assert observations_match(a, b)
    b[0, 0, 0] = 1.0 - b[0, 0, 0]
    assert not observations_match(a, b)


def test_observation_shape_and_constant_planes():
    ref, _ = _play_both(["e2e4"])  # Black to move, 1 ply played
    obs = ref.observation()
    assert obs.shape == (8, 8, 119) and obs.dtype == np.float32
    assert (obs[..., 112] == 1).all()  # colour plane: 1 when Black is to move
    assert np.allclose(obs[..., 113], 1 / 512)  # plies played / 512
    assert (obs[..., 114:118] == 1).all()  # castling rights: my Q, my K, opp Q, opp K
    assert (obs[..., 118] == 0).all()  # no-progress count / 100 (a pawn just moved)


def _outcome_after(fen, ucis=()):
    ref = ReferenceGame(fen)
    for uci in ucis:
        ref.push(chess.Move.from_uci(uci))
    return ref.outcome()


def test_game_in_progress_is_not_terminal():
    assert _outcome_after(None) == (False, 0.0)


def test_checkmate_is_a_loss_for_the_side_to_move():
    assert _outcome_after(None, ["f2f3", "e7e5", "g2g4", "d8h4"]) == (True, -1.0)


@pytest.mark.parametrize("fen", [
    "7k/5Q2/6K1/8/8/8/8/8 b - - 0 1",  # stalemate
    "8/8/4k3/8/8/3NK3/8/8 w - - 0 1",  # insufficient material: K+N vs K
    "8/2b5/4k3/8/8/2B1K3/8/8 w - - 0 1",  # insufficient material: both bishops on dark squares
])
def test_draw_positions(fen):
    assert _outcome_after(fen) == (True, 0.0)


def test_fifty_move_rule_ends_the_game_at_100_plies_without_progress():
    assert _outcome_after("8/8/4k3/8/8/4K3/4R3/8 w - - 99 80") == (False, 0.0)
    assert _outcome_after("8/8/4k3/8/8/4K3/4R3/8 w - - 99 80", ["e2e1"]) == (True, 0.0)


def test_threefold_repetition_ends_the_game():
    shuffle = ["g1f3", "g8f6", "f3g1", "f6g8"]
    assert _outcome_after(None, shuffle)[0] is False  # start position seen twice
    assert _outcome_after(None, shuffle * 2) == (True, 0.0)  # three times


def test_game_is_adjudicated_a_draw_at_the_ply_cap():
    ref = ReferenceGame(max_plies=4)
    for uci in ["e2e4", "e7e5", "g1f3", "b8c6"]:
        ref.push(chess.Move.from_uci(uci))
    assert ref.outcome() == (True, 0.0)
