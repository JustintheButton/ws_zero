import json

import jax
import numpy as np

from ws_zero.game.env import ChessEnv
from ws_zero.net import BN_LAYERS, apply, init_params


def _trained_looking_params():
    """Random weights with batch norm statistics and scales far from the identity, as after training."""
    params = init_params(jax.random.key(0))
    keys = iter(jax.random.split(jax.random.key(1), 4 * len(BN_LAYERS) + 4))
    for name in BN_LAYERS:
        shape = params[f"{name}/scale"].shape
        params[f"{name}/scale"] = 0.5 + jax.random.uniform(next(keys), shape)
        params[f"{name}/offset"] = 0.3 * jax.random.normal(next(keys), shape)
        params[f"{name}/mean"] = 0.3 * jax.random.normal(next(keys), shape)
        params[f"{name}/var"] = 0.5 + jax.random.uniform(next(keys), shape)
    for name in ("policy/out/w", "value/fc2/w"):  # zero at initialization; give the heads something to say
        params[name] = 0.1 * jax.random.normal(next(keys), params[name].shape)
    return params


def _observations():
    env = ChessEnv()
    state = env.init(6)
    obs = []
    key = jax.random.key(2)
    for _ in range(12):
        key, sub = jax.random.split(key)
        noise = jax.random.uniform(sub, state.legal_action_mask.shape)
        state = env.step(state, (noise * state.legal_action_mask).argmax(-1))
        obs.append(env.observe(state))
    return np.concatenate(obs)


def test_folding_batch_norm_into_the_convs_gives_the_same_network():
    from ws_zero.export import apply_folded, fold_batch_norm
    params = _trained_looking_params()
    obs = _observations()
    logits, value = apply(params, obs)
    folded_logits, folded_value = apply_folded(fold_batch_norm(params), obs)
    # float32 rounding through 13 layers differs by up to ~3e-5; a folding mistake would be off by ~0.1
    np.testing.assert_allclose(folded_logits, logits, atol=1e-4)
    np.testing.assert_allclose(folded_value, value, atol=1e-4)
    assert np.std(np.asarray(value)) > 0.01  # the test network says different things about different positions


def test_a_written_model_file_reads_back_exactly(tmp_path):
    from ws_zero.export import fold_batch_norm, read_model, write_model
    params = _trained_looking_params()
    write_model(tmp_path / "m", params, {"name": "test"})
    manifest = json.loads((tmp_path / "m.json").read_text())
    assert manifest["name"] == "test"
    back = read_model(tmp_path / "m")
    for name, array in fold_batch_norm(params).items():
        np.testing.assert_array_equal(back[name], np.asarray(array, np.float32))


def test_model_ids_that_contain_a_dot_keep_their_own_files(tmp_path):
    # Path.with_suffix would cut "random-seed0-1.5k" to "random-seed0-1", so two checkpoints would share one file.
    from ws_zero.export import read_model, write_model
    first, second = _trained_looking_params(), init_params(jax.random.key(3))
    write_model(tmp_path / "random-seed0-1.5k", first, {"name": "a"})
    write_model(tmp_path / "random-seed0-1.0k", second, {"name": "b"})
    assert sorted(p.name for p in tmp_path.iterdir()) == [
        "random-seed0-1.0k.bin", "random-seed0-1.0k.json", "random-seed0-1.5k.bin", "random-seed0-1.5k.json"]
    assert json.loads((tmp_path / "random-seed0-1.5k.json").read_text())["name"] == "a"
    from ws_zero.export import fold_batch_norm
    for model, params in [("random-seed0-1.5k", first), ("random-seed0-1.0k", second)]:
        back = read_model(tmp_path / model)
        for name, array in fold_batch_norm(params).items():
            np.testing.assert_array_equal(back[name], np.asarray(array, np.float32))
