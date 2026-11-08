import chess
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from ws_zero.game.moves import NUM_ACTIONS
from ws_zero.game.reference import ReferenceGame
from ws_zero.net import init_params, loss_fn
from ws_zero.train import make_optimizer, make_train_step


def _batch(n=8, seed=0):
    """n positions from one random game, one-hot policy targets on a random legal move, random results."""
    rng = np.random.default_rng(seed)
    ref, obs, masks, pis, zs = ReferenceGame(), [], [], [], []
    while len(obs) < n:
        legal = ref.legal_actions()
        mask = np.zeros(NUM_ACTIONS, dtype=bool)
        mask[legal] = True
        pi = np.zeros(NUM_ACTIONS, dtype=np.float32)
        pi[rng.choice(legal)] = 1.0
        obs.append(ref.observation()); masks.append(mask); pis.append(pi); zs.append(rng.choice([-1.0, 0.0, 1.0]))
        ref.push(rng.choice(list(ref.board.legal_moves)))
    return tuple(jnp.asarray(np.stack(a)) for a in (obs, masks, pis, np.array(zs, dtype=np.float32)))


def test_weight_decay_applies_to_weights_but_not_biases():
    params = {"layer/w": jnp.full((2, 2), 2.0), "layer/b": jnp.full((2,), 2.0)}
    opt = make_optimizer(learning_rate=1.0, momentum=0.0, weight_decay=0.1)
    zero_grads = jax.tree.map(jnp.zeros_like, params)
    updates, _ = opt.update(zero_grads, opt.init(params), params)
    np.testing.assert_allclose(updates["layer/w"], -0.2)  # -lr * wd * w
    np.testing.assert_allclose(updates["layer/b"], 0.0)


def test_momentum_matches_heavy_ball_sgd():
    # PyTorch-style SGD: buf = momentum * buf + g; w -= lr * buf.
    params = {"layer/b": jnp.zeros(1)}
    opt = make_optimizer(learning_rate=0.5, momentum=0.9, weight_decay=0.0)
    state, g = opt.init(params), {"layer/b": jnp.ones(1)}
    u1, state = opt.update(g, state, params)
    u2, state = opt.update(g, state, params)
    assert float(u1["layer/b"][0]) == pytest.approx(-0.5)
    assert float(u2["layer/b"][0]) == pytest.approx(-0.5 * 1.9)


def test_train_step_lowers_the_loss_on_a_fixed_batch():
    batch = _batch()
    params = init_params(jax.random.key(0))
    # LR 0.003, the smallest in the D1 grid: at 0.01, memorizing random results on 8 positions saturates the
    # value head (Phase 3 report).
    opt = make_optimizer(learning_rate=0.003)
    step = make_train_step(opt)
    state = opt.init(params)
    before = float(loss_fn(params, *batch)[0])
    for _ in range(20):
        params, state, loss, parts = step(params, state, batch)
    assert float(loss_fn(params, *batch)[0]) < 0.9 * before
    assert set(parts) == {"value", "policy"}
