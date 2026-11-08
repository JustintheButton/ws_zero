"""Overnight diagnostic: value-head saturation under longer training, with optional gradient clipping.

    python scripts/stability_phase3.py gate  <clip> <seeds> <steps>   # 256 positions, full batch, realistic targets
    python scripts/stability_phase3.py check <clip> <seeds> <steps>   # 16,384 positions, minibatch 1,024, held-out 2,048
clip = 0 means no clipping (the protocol). Prints, per seed: first step with >= 50% saturated values (if any),
final errors, and gradient-norm percentiles.
"""
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax

sys.path.insert(0, str(Path(__file__).parent))
import check_value_head as C  # noqa: E402
import gate_phase3 as G  # noqa: E402
from ws_zero.net import apply, init_params, loss_fn, masked_log_softmax, value_head_health  # noqa: E402
from ws_zero.seeding import derive_seed  # noqa: E402
from ws_zero.train import make_optimizer  # noqa: E402

mode, clip, seeds, steps = sys.argv[1], float(sys.argv[2]), [int(s) for s in sys.argv[3].split(",")], int(sys.argv[4])

if mode == "gate":
    data = {}
else:
    obs_all, masks_all = C.collect(16384 + 2048, derive_seed(0, "value-check-data"))
    test_obs = jnp.asarray(obs_all[16384:], dtype=jnp.float32)
    test_mask = jnp.asarray(masks_all[16384:])
    test_pi, test_z = C.targets(test_obs, test_mask)

for seed in seeds:
    if mode == "gate":
        (obs, mask, _, _), _ = G.random_positions(256, derive_seed(seed, "positions"))
        pi, z = C.targets(obs, mask)
        eval_obs, eval_mask, eval_pi, eval_z = obs, mask, pi, z
    else:
        eval_obs, eval_mask, eval_pi, eval_z = test_obs, test_mask, test_pi, test_z
    ent = -jnp.mean(jnp.sum(jnp.where(eval_pi > 0, eval_pi * jnp.log(jnp.where(eval_pi > 0, eval_pi, 1.0)), 0.0), -1))
    base = make_optimizer(0.003)
    opt = optax.chain(optax.clip_by_global_norm(clip), base) if clip > 0 else base
    params = init_params(jax.random.key(derive_seed(seed, "init")))
    state = opt.init(params)

    @jax.jit
    def step(params, state, batch):
        (loss, _), g = jax.value_and_grad(loss_fn, has_aux=True)(params, *batch)
        u, state = opt.update(g, state, params)
        return optax.apply_updates(params, u), state, optax.global_norm(g)

    @jax.jit
    def evaluate(params):
        logits, v = apply(params, eval_obs)
        kl = -jnp.mean(jnp.sum(eval_pi * masked_log_softmax(logits, eval_mask), -1)) - ent
        return jnp.mean((eval_z - v) ** 2), kl, value_head_health(params, eval_obs)["saturated"]

    rng = np.random.default_rng(derive_seed(seed, "batches"))
    norms, first_sat, best_mse = [], None, 9.0
    for i in range(steps + 1):
        if i % 50 == 0:
            mse, kl, sat = (float(x) for x in evaluate(params))
            best_mse = min(best_mse, mse)
            if first_sat is None and sat >= 0.5:
                first_sat = i
        if i < steps:
            if mode == "gate":
                batch = (obs, mask, pi, z)
            else:
                idx = rng.integers(0, 16384, 1024)
                o = jnp.asarray(obs_all[idx], dtype=jnp.float32)
                m = jnp.asarray(masks_all[idx])
                p, zz = C.targets(o, m)
                batch = (o, m, p, zz)
            params, state, gn = step(params, state, batch)
            norms.append(float(gn))
    q = np.percentile(norms, [50, 90, 99, 100])
    print(f"{mode} clip {clip:g} seed {seed} steps {steps}: saturated from step {first_sat} | final value error {mse:.4f} "
          f"(best {best_mse:.4f}) move KL {kl:.3f} saturated {sat:.2f} | grad norm p50 {q[0]:.1f} p90 {q[1]:.1f} "
          f"p99 {q[2]:.1f} max {q[3]:.1f}", flush=True)
