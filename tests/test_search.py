import chess
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from ws_zero.game.env import ChessEnv, state_from_fen
from ws_zero.game.moves import NUM_ACTIONS, move_to_action
from ws_zero.net import ILLEGAL_LOGIT
from ws_zero.search import make_recurrent_fn, make_root

ENV = ChessEnv()


def _batch(fen):
    return jax.tree.map(lambda x: x[None], state_from_fen(fen))


def _action(fen, uci):
    return jnp.array([move_to_action(chess.Board(fen), chess.Move.from_uci(uci))])


def _constant(value):
    """An evaluator with uniform move logits and a fixed value for the side to move."""
    def evaluate(params, obs):
        return jnp.zeros((obs.shape[0], NUM_ACTIONS)), jnp.full((obs.shape[0],), value)
    return evaluate


MATE_IN_ONE = "6k1/5ppp/8/8/8/8/5PPP/R5K1 w - - 0 1"  # Ra8#
STALEMATE_IN_ONE = "7k/8/5QK1/8/8/8/8/8 w - - 0 1"  # Qf7 stalemates
CHECKMATED = "R5k1/5ppp/8/8/8/8/5PPP/6K1 b - - 1 1"  # Black to move, mated


def test_root_uses_the_network_and_masks_illegal_moves():
    state = _batch(chess.STARTING_FEN)
    root = make_root(ENV, _constant(0.25), None, state)
    legal = np.asarray(state.legal_action_mask[0])
    assert legal.sum() == 20
    np.testing.assert_array_equal(np.asarray(root.prior_logits[0])[legal], 0.0)
    np.testing.assert_array_equal(np.asarray(root.prior_logits[0])[~legal], ILLEGAL_LOGIT)
    assert float(root.value[0]) == 0.25


def test_an_ordinary_move_hands_the_value_to_the_opponent_with_a_sign_flip():
    # The child's value is for the new side to move; discount -1 makes it count against the mover.
    recurrent = make_recurrent_fn(ENV, _constant(0.5))
    out, child = recurrent(None, None, _action(chess.STARTING_FEN, "e2e4"), _batch(chess.STARTING_FEN))
    assert float(out.reward[0]) == 0.0
    assert float(out.discount[0]) == -1.0
    assert float(out.value[0]) == 0.5
    legal = np.asarray(child.legal_action_mask[0])  # Black's replies
    assert legal.sum() == 20
    np.testing.assert_array_equal(np.asarray(out.prior_logits[0])[~legal], ILLEGAL_LOGIT)


def test_delivering_checkmate_rewards_the_mover_and_ends_the_line():
    recurrent = make_recurrent_fn(ENV, _constant(0.5))
    out, child = recurrent(None, None, _action(MATE_IN_ONE, "a1a8"), _batch(MATE_IN_ONE))
    assert bool(ENV.is_terminal(child)[0])
    assert float(out.reward[0]) == 1.0
    assert float(out.discount[0]) == 0.0
    assert float(out.value[0]) == 0.0  # a finished game has no value beyond its result


def test_a_move_into_stalemate_is_a_draw():
    recurrent = make_recurrent_fn(ENV, _constant(0.5))
    out, child = recurrent(None, None, _action(STALEMATE_IN_ONE, "f6f7"), _batch(STALEMATE_IN_ONE))
    assert bool(ENV.is_terminal(child)[0])
    assert float(out.reward[0]) == 0.0
    assert float(out.discount[0]) == 0.0


def test_stepping_past_the_end_of_a_game_changes_nothing():
    # The search can try to go below a finished position; that must not add a second result.
    recurrent = make_recurrent_fn(ENV, _constant(0.5))
    state = _batch(CHECKMATED)
    assert bool(ENV.is_terminal(state)[0])
    out, child = recurrent(None, None, jnp.array([0]), state)
    assert float(out.reward[0]) == 0.0
    assert float(out.discount[0]) == 0.0
    assert float(out.value[0]) == 0.0
    np.testing.assert_array_equal(np.asarray(child.board), np.asarray(state.board))  # same position


def test_the_recurrent_function_runs_on_a_batch_under_jit():
    recurrent = jax.jit(make_recurrent_fn(ENV, _constant(0.0)))
    fens = [chess.STARTING_FEN, MATE_IN_ONE]
    state = jax.tree.map(lambda *xs: jnp.stack(xs), *[state_from_fen(f) for f in fens])
    actions = jnp.concatenate([_action(chess.STARTING_FEN, "g1f3"), _action(MATE_IN_ONE, "a1a8")])
    out, _ = recurrent(None, jax.random.key(0), actions, state)
    np.testing.assert_array_equal(np.asarray(out.reward), [0.0, 1.0])
    np.testing.assert_array_equal(np.asarray(out.discount), [-1.0, 0.0])
    assert out.prior_logits.shape == (2, NUM_ACTIONS)


@pytest.mark.parametrize("fen", [MATE_IN_ONE, STALEMATE_IN_ONE])
def test_the_test_positions_are_what_they_claim(fen):
    board = chess.Board(fen)
    move = {MATE_IN_ONE: "a1a8", STALEMATE_IN_ONE: "f6f7"}[fen]
    board.push(chess.Move.from_uci(move))
    assert board.is_checkmate() if fen == MATE_IN_ONE else board.is_stalemate()


# ---------- AlphaZero's value term (untried moves score 0, "equal"; decision 2026-11-08) ----------

def _search(fen, evaluate, simulations, seed=0):
    from ws_zero.search import run_search
    state = _batch(fen)
    run = jax.jit(lambda key, s: run_search(ENV, evaluate, None, key, s, simulations, dirichlet_fraction=0.0))
    return run(jax.random.key(seed), state)


def test_untried_moves_count_as_equal_so_every_root_move_is_tried():
    # Uniform priors and value 0 everywhere: an untried move ("equal", 0) is as good as a tried one, and its larger
    # exploration bonus makes the search try every move before repeating any. Counting untried moves as losses
    # (the pseudocode read literally) would keep the search on its first move.
    out = _search(chess.STARTING_FEN, _constant(0.0), 40)
    visits = np.asarray(out.search_tree.summary().visit_counts[0])
    legal = np.asarray(_batch(chess.STARTING_FEN).legal_action_mask[0])
    assert (visits[legal] >= 1).all()
    assert visits[~legal].sum() == 0


def test_value_term_is_the_mean_value_zero_when_untried_and_never_an_illegal_move():
    from ws_zero.search import alphazero_qtransform
    out = _search(chess.STARTING_FEN, _constant(0.3), 8)  # 8 simulations: most of the 20 moves stay untried
    tree = jax.tree.map(lambda x: x[0], out.search_tree)  # game 0
    score = np.asarray(alphazero_qtransform(tree, tree.ROOT_INDEX))
    visits = np.asarray(tree.children_visits[tree.ROOT_INDEX])
    legal = np.asarray(_batch(chess.STARTING_FEN).legal_action_mask[0])
    q = np.asarray(tree.qvalues(tree.ROOT_INDEX))
    tried, untried = legal & (visits > 0), legal & (visits == 0)
    assert tried.any() and untried.any()
    np.testing.assert_allclose(score[tried], q[tried])
    np.testing.assert_allclose(score[tried], -0.3)  # each reply position is worth +0.3 to the opponent
    np.testing.assert_array_equal(score[untried], 0.0)
    assert np.isneginf(score[~legal]).all()


def test_the_search_never_plays_an_illegal_move_below_the_root():
    # Every side to move thinks it is winning (+0.9), so every tried move scores -0.9 for its mover. Once a position's
    # few legal moves are all tried, an illegal move scored like an untried one (0) would beat them; mctx blocks
    # illegal moves only at the root, so the value term must block them everywhere else.
    fen = "k7/8/8/8/8/8/1P6/K7 w - - 0 1"  # 4 legal moves for White, 3 for Black
    out = _search(fen, _constant(0.9), 200)
    tree = out.search_tree
    visits = np.asarray(tree.children_visits[0])  # (nodes, moves)
    legal = np.asarray(tree.embeddings.legal_action_mask[0])  # each node's legal moves
    live = legal.any(axis=1)  # nodes that are positions with moves (not finished games or empty slots)
    assert (visits[live] > 0).sum() > 20  # the search went well below the root
    assert not np.any((visits > 0) & ~legal & live[:, None])
