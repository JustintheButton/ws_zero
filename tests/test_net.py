import chess
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from ws_zero.game.moves import NUM_ACTIONS, move_to_action
from ws_zero.game.reference import ReferenceGame
from ws_zero.net import (BN_EPS, BN_LAYERS, BN_MOMENTUM, PARAM_SHAPES, STAT_NAMES, ZERO_INIT, apply, apply_train,
                         batch_norm, blank_params, init_params, loss_fn, masked_log_softmax, num_params,
                         policy_to_actions)


def _obs_and_mask(moves=()):
    ref = ReferenceGame()
    for uci in moves:
        ref.push(chess.Move.from_uci(uci))
    mask = np.zeros(NUM_ACTIONS, dtype=bool)
    mask[ref.legal_actions()] = True
    return ref, jnp.asarray(ref.observation())[None], jnp.asarray(mask)[None]


def test_parameter_count_matches_the_protocol():
    trainable = {k: s for k, s in PARAM_SHAPES.items() if k not in STAT_NAMES}
    assert num_params(trainable) == 571_276
    assert num_params({k: PARAM_SHAPES[k] for k in STAT_NAMES}) == 1_794  # running means and variances


def test_batch_norm_follows_every_conv_except_the_policy_output():
    assert BN_LAYERS == ["stem/bn"] + [f"block{i}/bn{j}" for i in range(6) for j in (1, 2)] + ["policy/bn", "value/bn"]
    for conv in ["stem", "block0/conv1", "block5/conv2", "policy/conv", "value/conv"]:
        assert f"{conv}/b" not in PARAM_SHAPES  # the batch norm offset plays the bias's role
    assert "policy/out/b" in PARAM_SHAPES


def test_batch_norm_in_training_uses_batch_statistics_and_updates_the_running_ones():
    x = jax.random.normal(jax.random.key(0), (16, 8, 8, 3)) * jnp.array([1.0, 2.0, 3.0]) + jnp.array([5.0, -1.0, 0.0])
    scale, offset = jnp.array([1.0, 2.0, 1.0]), jnp.array([0.0, 0.0, 3.0])
    mean, var = jnp.array([1.0, 1.0, 1.0]), jnp.array([4.0, 4.0, 4.0])
    y, new_mean, new_var = batch_norm(x, scale, offset, mean, var, train=True)
    np.testing.assert_allclose(y.mean(axis=(0, 1, 2)), offset, atol=1e-4)
    np.testing.assert_allclose(y.std(axis=(0, 1, 2)), scale, rtol=1e-3)
    batch_mean, batch_var = x.mean(axis=(0, 1, 2)), x.var(axis=(0, 1, 2))
    np.testing.assert_allclose(new_mean, (1 - BN_MOMENTUM) * mean + BN_MOMENTUM * batch_mean, rtol=1e-5)
    np.testing.assert_allclose(new_var, (1 - BN_MOMENTUM) * var + BN_MOMENTUM * batch_var, rtol=1e-5)


def test_batch_norm_in_play_uses_the_running_statistics():
    x = jax.random.normal(jax.random.key(0), (4, 8, 8, 2))
    scale, offset, mean, var = jnp.array([2.0, 1.0]), jnp.array([0.5, 0.0]), jnp.array([1.0, -1.0]), jnp.array([4.0, 1.0])
    y, new_mean, new_var = batch_norm(x, scale, offset, mean, var, train=False)
    np.testing.assert_allclose(y, (x - mean) / jnp.sqrt(var + BN_EPS) * scale + offset, rtol=1e-6)
    np.testing.assert_array_equal(new_mean, mean)
    np.testing.assert_array_equal(new_var, var)


def test_training_forward_returns_new_running_statistics_for_every_layer():
    params = init_params(jax.random.key(0))
    obs = _positions()
    logits, v, stats = apply_train(params, obs)
    assert set(stats) == STAT_NAMES
    assert not np.allclose(stats["stem/bn/mean"], params["stem/bn/mean"])
    np.testing.assert_allclose(apply(params, obs)[0], logits, atol=1e-6)  # policy output is zero-initialized


def test_running_statistics_get_no_gradient():
    _, obs, mask = _obs_and_mask(["e2e4"])
    params = init_params(jax.random.key(0))
    pi = mask.astype(jnp.float32) / mask.sum()
    grads = jax.grad(lambda p: loss_fn(p, obs, mask, pi, jnp.array([1.0]))[0])(params)
    for name in STAT_NAMES:
        assert not np.any(np.asarray(grads[name])), name
    assert np.any(np.asarray(grads["value/fc2/w"]))


def test_init_has_the_protocol_shapes():
    params = init_params(jax.random.key(0))
    assert {k: v.shape for k, v in params.items()} == PARAM_SHAPES


def test_init_zeroes_biases_and_the_last_layer_of_each_branch():
    params = init_params(jax.random.key(0))
    for name, value in params.items():
        value = np.asarray(value)
        if name in ZERO_INIT or name.endswith(("/b", "/offset", "/mean")):
            assert not np.any(value), name
        elif name.endswith(("/scale", "/var")):
            np.testing.assert_array_equal(value, 1.0, err_msg=name)
        else:
            assert np.any(value), name
    # Each residual branch ends in a batch norm whose scale starts at 0, so every block starts as the identity.
    assert {f"block{i}/bn2/scale" for i in range(6)} | {"policy/out/w", "value/fc2/w"} == ZERO_INIT


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
    params = blank_params()  # every weight 0, every batch norm the identity
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
        # rtol: each identity batch norm divides by sqrt(1 + eps).
        np.testing.assert_allclose(p[doubles], np.e / (n_doubles * np.e + n_legal - n_doubles), rtol=1e-4)


def test_hand_written_value_reads_a_square_in_row_major_order():
    # Value head input is the 8 x 8 map flattened row by row: square (row, column) is feature row * 8 + column.
    params = blank_params()
    params["stem/w"] = params["stem/w"].at[1, 1, 0, 0].set(1.0)  # my pawns
    params["value/conv/w"] = params["value/conv/w"].at[0, 0, 0, 0].set(1.0)
    params["value/fc1/w"] = params["value/fc1/w"].at[6 * 8 + 4, 0].set(1.0)  # row 6, column 4: e2 for White
    params["value/fc2/w"] = params["value/fc2/w"].at[0, 0].set(1.0)
    _, obs, _ = _obs_and_mask([])
    assert float(apply(params, obs)[1][0]) == pytest.approx(np.tanh(1.0), rel=1e-4)
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


def test_value_conv_is_followed_by_batch_norm_and_relu_as_in_alphazero():
    # With batch norm restored (2026-11-08) the value head is AlphaZero's again: negative values stop at the ReLU.
    params = blank_params()
    params["stem/w"] = params["stem/w"].at[1, 1, 0, 0].set(1.0)  # my pawns
    params["value/conv/w"] = params["value/conv/w"].at[0, 0, 0, 0].set(-1.0)  # -1 where my pawns stand
    params["value/fc1/w"] = params["value/fc1/w"].at[6 * 8 + 4, 0].set(-1.0)  # e2 for White, sign flipped back
    params["value/fc2/w"] = params["value/fc2/w"].at[0, 0].set(1.0)
    _, obs, _ = _obs_and_mask([])
    assert float(apply(params, obs)[1][0]) == 0.0
    params["value/bn/offset"] = jnp.array([2.0])  # shift the value map up: -1 + 2 = 1 on my pawns, 2 elsewhere
    params["value/fc1/w"] = params["value/fc1/w"].at[6 * 8 + 4, 0].set(1.0)
    assert float(apply(params, obs)[1][0]) == pytest.approx(np.tanh(1.0), rel=1e-4)


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
    assert 0.0 < health["fc1_active"] < 1.0 and 0.0 < health["conv_active"] < 1.0
    params["value/fc2/w"] = jnp.full(PARAM_SHAPES["value/fc2/w"], 1e3)
    params["value/fc1/b"] = jnp.full(PARAM_SHAPES["value/fc1/b"], 1.0)
    assert float(value_head_health(params, obs)["saturated"]) == 1.0
    params["value/fc1/b"] = jnp.full(PARAM_SHAPES["value/fc1/b"], -1e3)  # every unit off: no gradient reaches it
    assert float(value_head_health(params, obs)["fc1_active"]) == 0.0
    params["value/bn/offset"] = jnp.array([-1e3])  # the value map's ReLU off on every square (the original failure)
    assert float(value_head_health(params, obs)["conv_active"]) == 0.0
