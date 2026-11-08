import chess
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from ws_zero.game.moves import NUM_ACTIONS, move_to_action
from ws_zero.game.reference import ReferenceGame
from ws_zero.net import (PARAM_SHAPES, ZERO_INIT, apply, init_params, loss_fn, masked_log_softmax, num_params,
                         policy_to_actions)


def _obs_and_mask(moves=()):
    ref = ReferenceGame()
    for uci in moves:
        ref.push(chess.Move.from_uci(uci))
    mask = np.zeros(NUM_ACTIONS, dtype=bool)
    mask[ref.legal_actions()] = True
    return ref, jnp.asarray(ref.observation())[None], jnp.asarray(mask)[None]


def _zeros():
    return {name: jnp.zeros(shape) for name, shape in PARAM_SHAPES.items()}


def test_parameter_count_matches_the_protocol():
    assert num_params(PARAM_SHAPES) == 570_379


def test_init_has_the_protocol_shapes():
    params = init_params(jax.random.key(0))
    assert {k: v.shape for k, v in params.items()} == PARAM_SHAPES


def test_init_zeroes_biases_and_the_last_layer_of_each_branch():
    params = init_params(jax.random.key(0))
    for name, value in params.items():
        if name.endswith("/b") or name in ZERO_INIT:
            assert not np.any(np.asarray(value)), name
        else:
            assert np.any(np.asarray(value)), name
    assert "block5/conv2/w" in ZERO_INIT and "policy/out/w" in ZERO_INIT and "value/fc2/w" in ZERO_INIT


def test_init_is_he_normal():
    params = init_params(jax.random.key(0))
    stem = np.asarray(params["stem/w"])  # fan-in 3 * 3 * 119
    assert stem.std() == pytest.approx(np.sqrt(2 / (3 * 3 * 119)), rel=0.03)
    assert abs(stem.mean()) < 0.003
    fc1 = np.asarray(params["value/fc1/w"])  # fan-in 64
    assert fc1.std() == pytest.approx(np.sqrt(2 / 64), rel=0.05)


def test_init_depends_on_the_key():
    a, b = init_params(jax.random.key(0)), init_params(jax.random.key(1))
    assert not np.array_equal(a["stem/w"], b["stem/w"])
    np.testing.assert_array_equal(a["stem/w"], init_params(jax.random.key(0))["stem/w"])


def test_random_init_gives_a_uniform_policy_and_zero_value():
    _, obs, mask = _obs_and_mask(["e2e4", "c7c5"])
    logits, value = apply(init_params(jax.random.key(0)), obs)
    assert logits.shape == (1, NUM_ACTIONS) and value.shape == (1,)
    p = np.exp(np.asarray(masked_log_softmax(logits, mask)))[0]
    legal = np.asarray(mask[0])
    np.testing.assert_allclose(p[legal], 1 / legal.sum(), rtol=1e-6)
    assert p[~legal].max() < 1e-30
    assert float(value[0]) == 0.0


def test_policy_layout_maps_each_square_and_plane_to_its_action():
    # Observation square (row, column) and move type `plane` belong to action (column * 8 + 7 - row) * 73 + plane.
    x = np.zeros((1, 8, 8, 73), dtype=np.float32)
    x[0, 6, 4, 17] = 1.0  # e2, two squares forward, from White's side
    assert int(np.argmax(policy_to_actions(jnp.asarray(x))[0])) == 2426 == move_to_action(
        chess.Board(), chess.Move.from_uci("e2e4"))
    rng = np.random.default_rng(0)
    for _ in range(20):
        r, c, p = rng.integers(8), rng.integers(8), rng.integers(73)
        x = np.zeros((1, 8, 8, 73), dtype=np.float32)
        x[0, r, c, p] = 1.0
        assert int(np.argmax(policy_to_actions(jnp.asarray(x))[0])) == (c * 8 + 7 - r) * 73 + p


def test_hand_written_policy_prefers_double_pawn_pushes():
    # A three-weight network: stem copies "my pawns" to channel 0, the policy head turns it into
    # "move two squares forward" (plane 17) from every square holding my pawn.
    params = _zeros()
    params["stem/w"] = params["stem/w"].at[1, 1, 0, 0].set(1.0)
    params["policy/conv/w"] = params["policy/conv/w"].at[1, 1, 0, 0].set(1.0)
    params["policy/out/w"] = params["policy/out/w"].at[0, 0, 0, 17].set(1.0)
    for moves in ([], ["e2e4"]):  # White to move, then Black to move (board flipped)
        ref, obs, mask = _obs_and_mask(moves)
        logits, _ = apply(params, obs)
        p = np.exp(np.asarray(masked_log_softmax(logits, mask)))[0]
        doubles = [move_to_action(ref.board, m) for m in ref.board.legal_moves
                   if chess.square_distance(m.from_square, m.to_square) == 2
                   and ref.board.piece_type_at(m.from_square) == chess.PAWN]
        n_doubles, n_legal = len(doubles), int(mask.sum())
        assert n_doubles == 8
        np.testing.assert_allclose(p[doubles], np.e / (n_doubles * np.e + n_legal - n_doubles), rtol=1e-5)


def test_hand_written_value_reads_a_square_in_row_major_order():
    # Value head input is the 8 x 8 map flattened row by row: square (row, column) is feature row * 8 + column.
    params = _zeros()
    params["stem/w"] = params["stem/w"].at[1, 1, 0, 0].set(1.0)  # my pawns
    params["value/conv/w"] = params["value/conv/w"].at[0, 0, 0, 0].set(1.0)
    params["value/fc1/w"] = params["value/fc1/w"].at[6 * 8 + 4, 0].set(1.0)  # row 6, column 4: e2 for White
    params["value/fc2/w"] = params["value/fc2/w"].at[0, 0].set(1.0)
    _, obs, _ = _obs_and_mask([])
    assert float(apply(params, obs)[1][0]) == pytest.approx(np.tanh(1.0))
    _, obs, _ = _obs_and_mask(["e2e4", "e7e5"])
    assert float(apply(params, obs)[1][0]) == 0.0


def test_value_is_bounded_by_tanh():
    params = init_params(jax.random.key(0))
    params["value/fc2/w"] = jnp.full(PARAM_SHAPES["value/fc2/w"], 100.0)
    _, obs, _ = _obs_and_mask([])
    v = float(apply(params, obs)[1][0])
    assert -1.0 <= v <= 1.0


def test_loss_is_value_error_plus_policy_cross_entropy():
    _, obs, mask = _obs_and_mask([])
    params = init_params(jax.random.key(0))  # uniform policy over 20 moves, value 0
    pi = mask.astype(jnp.float32) / mask.sum()
    loss, parts = loss_fn(params, obs, mask, pi, jnp.array([1.0]))
    assert float(parts["value"]) == pytest.approx(1.0)  # (1 - 0)^2
    assert float(parts["policy"]) == pytest.approx(np.log(20), rel=1e-5)
    assert float(loss) == pytest.approx(1.0 + np.log(20), rel=1e-5)


def test_value_conv_feeds_the_value_layer_without_a_relu():
    # Amendment 2026-11-08: the value head's 1x1 conv is linear, so negative values reach the 256-unit layer.
    params = _zeros()
    params["stem/w"] = params["stem/w"].at[1, 1, 0, 0].set(1.0)  # my pawns
    params["value/conv/w"] = params["value/conv/w"].at[0, 0, 0, 0].set(-1.0)  # -1 where my pawns stand
    params["value/fc1/w"] = params["value/fc1/w"].at[6 * 8 + 4, 0].set(-1.0)  # e2 for White, sign flipped back
    params["value/fc2/w"] = params["value/fc2/w"].at[0, 0].set(1.0)
    _, obs, _ = _obs_and_mask([])
    assert float(apply(params, obs)[1][0]) == pytest.approx(np.tanh(1.0))


def _positions(n=12):
    ref, obs = ReferenceGame(), []
    rng = np.random.default_rng(0)
    for _ in range(n):
        obs.append(ref.observation())
        ref.push(rng.choice(list(ref.board.legal_moves)))
    return jnp.asarray(np.stack(obs))


def test_value_head_health_reports_a_constant_a_saturated_and_a_dead_value_head():
    from ws_zero.net import value_head_health

    obs = _positions()
    params = init_params(jax.random.key(0))
    health = {k: float(v) for k, v in value_head_health(params, obs).items()}
    assert health["value_std"] == 0.0 and health["saturated"] == 0.0  # zero last layer: value 0 everywhere
    assert 0.0 < health["fc1_active"] < 1.0
    params["value/fc2/w"] = jnp.full(PARAM_SHAPES["value/fc2/w"], 1e3)
    params["value/fc1/b"] = jnp.full(PARAM_SHAPES["value/fc1/b"], 1.0)
    assert float(value_head_health(params, obs)["saturated"]) == 1.0
    params["value/fc1/b"] = jnp.full(PARAM_SHAPES["value/fc1/b"], -1e3)  # every unit off: no gradient reaches it
    assert float(value_head_health(params, obs)["fc1_active"]) == 0.0
