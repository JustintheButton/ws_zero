import re

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


# ---------- The value term: untried moves score the position's own value (decision 2026-11-08, revised) ----------

def _search(fen, evaluate, simulations, seed=0):
    from ws_zero.search import run_search
    state = _batch(fen)
    run = jax.jit(lambda key, s: run_search(ENV, evaluate, None, key, s, simulations, dirichlet_fraction=0.0))
    return run(jax.random.key(seed), state)


def test_in_a_level_position_every_root_move_is_tried():
    # Uniform priors and value 0 everywhere: an untried move (the position's value, 0) is as good as a tried one, and
    # its larger exploration bonus makes the search try every move before repeating any. Counting untried moves as
    # losses (the pseudocode read literally) would keep the search on its first move.
    out = _search(chess.STARTING_FEN, _constant(0.0), 40)
    visits = np.asarray(out.search_tree.summary().visit_counts[0])
    legal = np.asarray(_batch(chess.STARTING_FEN).legal_action_mask[0])
    assert (visits[legal] >= 1).all()
    assert visits[~legal].sum() == 0


def test_value_term_is_the_mean_value_the_position_value_when_untried_and_never_an_illegal_move():
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
    root_value = float(tree.node_values[tree.ROOT_INDEX])  # (0.3 + 8 x -0.3) / 9, from White's point of view
    np.testing.assert_allclose(root_value, (0.3 - 8 * 0.3) / 9, rtol=1e-5)
    np.testing.assert_allclose(score[untried], root_value)
    assert np.isneginf(score[~legal]).all()


def test_when_ahead_every_root_move_is_still_tried_and_the_mate_is_found():
    # A rook up, every tried move scores about +0.85. Scored as "equal" (0), untried moves looked worse than any
    # tried one, and the search spent 170+ of 200 simulations on its first quiet move without trying the mate
    # (Phase 4 gate, first run). Scored as the position's own value, they are tried.
    out = _search(MATE_IN_ONE, _material, 200)
    visits = np.asarray(out.search_tree.summary().visit_counts[0])
    legal = np.asarray(_batch(MATE_IN_ONE).legal_action_mask[0])
    assert (visits[legal] >= 1).all()
    assert int(np.argmax(visits)) == int(_action(MATE_IN_ONE, "a1a8")[0])


def test_the_tree_never_grows_more_than_one_move_below_a_finished_game():
    # Nothing below a finished game counts. With every move below one masked as illegal, each return visit to Ra8#
    # took the same slot one move deeper: in the Phase 4 gate the tree became a chain over 100 moves deep below the
    # mate, and every simulation walked it. Below a finished game the visits must spread out instead.
    out = _search(MATE_IN_ONE, _material, 200)
    tree = out.search_tree
    parents = np.asarray(tree.parents[0])
    over = np.asarray(ENV.is_terminal(jax.tree.map(lambda x: x[0], tree.embeddings)))
    has_parent = parents >= 0
    below = has_parent & over[np.where(has_parent, parents, 0)]  # nodes hanging below a finished game
    assert below.sum() > 50  # the search kept returning to the mate
    assert not below[parents[has_parent]].any()  # and none of those nodes has a node below it


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


# ---------- Move choice (protocol §3) ----------

VISITS = jnp.array([[1.0, 3.0, 0.0, 6.0]])


def test_self_play_samples_early_moves_in_proportion_to_visits():
    from ws_zero.search import choose_moves
    keys = jax.random.split(jax.random.key(0), 4000)
    picks = jax.vmap(lambda k: choose_moves(k, VISITS, jnp.array([0]), sample=True)[0])(keys)
    freq = np.bincount(np.asarray(picks), minlength=4) / 4000
    np.testing.assert_allclose(freq, [0.1, 0.3, 0.0, 0.6], atol=0.03)
    assert freq[2] == 0.0  # a move the search never visited is never played


def test_self_play_plays_the_most_visited_move_from_ply_30():
    from ws_zero.search import SAMPLING_PLIES, choose_moves
    assert SAMPLING_PLIES == 30
    visits, plies = jnp.repeat(VISITS, 3, axis=0), jnp.array([29, 30, 200])
    picks = np.stack([np.asarray(choose_moves(jax.random.key(i), visits, plies, sample=True)) for i in range(60)])
    assert (picks[:, 1] == 3).all() and (picks[:, 2] == 3).all()
    assert len(set(picks[:, 0].tolist())) > 1  # ply 29 (the 30th ply) is still sampled


def test_evaluation_plays_the_most_visited_move_from_the_first_ply():
    from ws_zero.search import choose_moves
    assert all(int(choose_moves(jax.random.key(i), VISITS, jnp.array([0]), sample=False)[0]) == 3 for i in range(20))


def _root_prior(out):
    return np.asarray(jax.nn.softmax(out.search_tree.children_prior_logits[0, out.search_tree.ROOT_INDEX]))


def test_root_noise_is_added_in_self_play_only():
    from ws_zero.search import search_moves
    state = _batch(chess.STARTING_FEN)
    legal = np.asarray(state.legal_action_mask[0])
    run = jax.jit(lambda key, self_play: search_moves(ENV, _constant(0.0), None, key, state, 16, self_play),
                  static_argnums=1)
    _, played = run(jax.random.key(0), True)
    _, evaluated = run(jax.random.key(0), False)
    np.testing.assert_allclose(_root_prior(evaluated)[legal], 1 / 20, rtol=1e-5)  # the network's own prior
    assert np.abs(_root_prior(played)[legal] - 1 / 20).max() > 0.01  # Dirichlet noise mixed in
    np.testing.assert_allclose(_root_prior(played)[legal].sum(), 1.0, rtol=1e-5)
    assert _root_prior(played)[~legal].max() < 1e-30


def test_chosen_moves_are_legal_and_a_finished_game_in_the_batch_is_harmless():
    from ws_zero.search import search_moves
    fens = [chess.STARTING_FEN, CHECKMATED, MATE_IN_ONE]
    state = jax.tree.map(lambda *xs: jnp.stack(xs), *[state_from_fen(f) for f in fens])
    actions, out = jax.jit(lambda key: search_moves(ENV, _constant(0.1), None, key, state, 32, True))(
        jax.random.key(1))
    tree = out.search_tree
    for values in (tree.node_values, tree.children_values, tree.children_prior_logits):
        assert not np.isnan(np.asarray(values)).any()
    legal = np.asarray(state.legal_action_mask)
    assert legal[0, int(actions[0])] and legal[2, int(actions[2])]
    after = ENV.step(state, actions)
    np.testing.assert_array_equal(np.asarray(after.board[1]), np.asarray(state.board[1]))  # finished: unchanged


def test_root_noise_is_dirichlet_over_the_legal_moves_mixed_in_at_a_quarter():
    # AlphaZero: P = 0.75 p + 0.25 eta, eta ~ Dirichlet(0.3) over the legal moves only. mctx's own noise is spread
    # over all 4,672 moves, so in chess almost none of it would reach the legal ones.
    from ws_zero.search import add_root_noise
    state = _batch(chess.STARTING_FEN)
    legal = state.legal_action_mask
    logits = jnp.where(legal, jnp.arange(NUM_ACTIONS) % 7 * 0.1, ILLEGAL_LOGIT)  # a non-uniform prior
    p = np.asarray(jax.nn.softmax(logits))[0]
    keys = jax.random.split(jax.random.key(0), 3000)
    mixed = np.asarray(jax.vmap(lambda k: jax.nn.softmax(add_root_noise(k, logits, legal, 0.25))[0])(keys))
    lm = np.asarray(legal[0])
    eta = (mixed[:, lm] - 0.75 * p[lm]) / 0.25  # the noise part, per draw
    assert (eta > -1e-5).all() and np.allclose(eta.sum(axis=1), 1.0, atol=1e-4)
    assert mixed[:, ~lm].max() < 1e-30
    n, alpha = 20, 0.3
    np.testing.assert_allclose(eta.mean(axis=0), 1 / n, atol=0.01)
    dirichlet_var = (1 / n) * (1 - 1 / n) / (n * alpha + 1)
    np.testing.assert_allclose(eta.var(axis=0).mean(), dirichlet_var, rtol=0.15)


# ---------- The sign flip at every level, and agreement with a plain-Python search ----------

PIECE_VALUES = np.array([1, 3, 3, 5, 9, 0], dtype=np.float32)  # P N B R Q K
CENTRE = np.array([[(7 - abs(2 * r - 7) + 7 - abs(2 * c - 7)) / 14 for c in range(8)] for r in range(8)],
                  dtype=np.float32)
ITALIAN = "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3"
SCHOLARS = "r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4"  # Qxf7#
PAWN_ENDING = "8/5k2/8/3p4/3P4/2K5/8/8 w - - 0 1"


def _material(params, obs):
    """A test network: every move a slightly different prior (no ties), and a value for the side to move from
    material and how central the pieces are."""
    pv, centre = jnp.asarray(PIECE_VALUES), jnp.asarray(CENTRE)
    mine = jnp.einsum("bhwp,p->b", obs[..., 0:6], pv)
    theirs = jnp.einsum("bhwp,p->b", obs[..., 6:12], pv)
    central = jnp.einsum("bhwp,hw->b", obs[..., 0:6], centre) - jnp.einsum("bhwp,hw->b", obs[..., 6:12], centre)
    logits = jnp.broadcast_to(0.5 * jnp.sin(1.7 * jnp.arange(NUM_ACTIONS)), (obs.shape[0], NUM_ACTIONS))
    return logits, jnp.tanh((mine - theirs) / 4 + 0.05 * central)


def test_values_flip_sign_at_every_level_of_the_tree():
    # Each position's recorded value, from its side to move, times its visits = its own evaluation + the sum over
    # its moves of visits x move value, and a move's value = reward - (the opponent's value of the next position),
    # or just the reward when the move ends the game. Checked at every node of a real search (a pawn ending: few
    # moves per position, so the tree goes deep).
    out = _search(PAWN_ENDING, _material, 300)
    tree = jax.tree.map(lambda x: x[0], out.search_tree)
    visits = np.asarray(tree.node_visits)
    live = visits > 0
    finished = np.asarray(ENV.is_terminal(tree.embeddings))
    own = np.where(finished, 0.0, np.asarray(_material(None, ENV.observe(tree.embeddings))[1]))
    values, kids = np.asarray(tree.node_values), np.asarray(tree.children_index)
    child_visits, rewards = np.asarray(tree.children_visits), np.asarray(tree.children_rewards)
    discounts = np.asarray(tree.children_discounts)
    parents, depth = np.asarray(tree.parents), np.zeros(len(visits), dtype=int)
    for i in np.flatnonzero(live):
        tried = child_visits[i] > 0
        q = rewards[i] + np.where(finished[kids[i]], 0.0, -1.0) * values[kids[i]]
        np.testing.assert_allclose(values[i] * visits[i], own[i] + (child_visits[i] * q)[tried].sum(), atol=1e-4)
        np.testing.assert_allclose(np.asarray(tree.qvalues(i))[tried], q[tried], atol=1e-6)
        np.testing.assert_array_equal(discounts[i][tried], np.where(finished[kids[i]], 0.0, -1.0)[tried])
        if i != tree.ROOT_INDEX:
            depth[i] = depth[parents[i]] + 1
    assert depth.max() >= 4  # the check covered several levels


@pytest.mark.parametrize("fen", [chess.STARTING_FEN, ITALIAN, SCHOLARS, MATE_IN_ONE, PAWN_ENDING])
def test_the_search_spends_its_simulations_exactly_like_a_plain_python_search(fen):
    from reference_search import reference_search, root_visits
    simulations = 100
    out = _search(fen, _material, simulations)
    evaluate = jax.jit(_material)

    def evaluate_one(obs):
        logits, value = evaluate(None, jnp.asarray(obs)[None])
        return np.asarray(logits[0], dtype=np.float64), float(value[0])

    reference = reference_search(fen, evaluate_one, simulations)
    np.testing.assert_array_equal(np.asarray(out.search_tree.summary().visit_counts[0]),
                                  root_visits(reference, NUM_ACTIONS))


# ---------- The backup (2026-11-09): our own, one entry per game per step ----------

BACKUP_FENS = [chess.STARTING_FEN, ITALIAN, SCHOLARS, PAWN_ENDING]


def _batch_of(fens):
    return jax.tree.map(lambda *xs: jnp.stack(xs), *[state_from_fen(f) for f in fens])


def _search_batch(evaluate, simulations, fens=BACKUP_FENS):
    from ws_zero.search import run_search
    state = _batch_of(fens)
    run = jax.jit(lambda key: run_search(ENV, evaluate, None, key, state, simulations, dirichlet_fraction=0.0))
    return run(jax.random.key(0))


def test_a_backup_step_never_copies_a_whole_tree_table():
    # mctx's backup is one game's loop, batched over the games with vmap. The games climb paths of different lengths,
    # so vmap turns every step into "keep the new tree or the old one?" over the whole of children_visits and
    # children_values ([games, simulations + 1, 4672]): cost ~ games x simulations^2 x 4672, most of a move's time
    # at 100 games x 800. The compiled search must contain no such select.
    from ws_zero.search import run_search
    games, simulations = 2, 8
    state = _batch_of(BACKUP_FENS[:games])
    run = jax.jit(lambda key: run_search(ENV, _constant(0.0), None, key, state, simulations, dirichlet_fraction=0.0))
    hlo = run.lower(jax.random.key(0)).compile().as_text()
    whole_table = re.compile(rf"(?:f32|s32)\[{games},{simulations + 1},{NUM_ACTIONS}\]\S* select\(")
    assert whole_table.findall(hlo) == []


def test_the_backup_builds_exactly_the_tree_mctx_builds(monkeypatch):
    # Several games at once whose backups climb paths of different lengths: the case where mctx's version and ours
    # could differ. Every number in the tree, the positions stored in it included, must match exactly.
    from mctx._src import search as mctx_search

    from ws_zero.search import MCTX_BACKWARD
    ours = _search_batch(_material, 100).search_tree
    monkeypatch.setattr(mctx_search, "backward", MCTX_BACKWARD)
    theirs = _search_batch(_material, 100).search_tree

    parents = np.asarray(ours.parents)
    depth = np.zeros(parents.shape, dtype=int)
    for i in range(1, parents.shape[1]):  # a node's parent always has a smaller index
        depth[:, i] = depth[np.arange(parents.shape[0]), parents[:, i]] + 1
    assert len(set(depth.max(axis=1))) > 1  # the games' deepest backups really differ in length

    leaves, _ = jax.tree_util.tree_flatten_with_path(ours)
    for (path, a), b in zip(leaves, jax.tree.leaves(theirs), strict=True):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b), err_msg=jax.tree_util.keystr(path))
