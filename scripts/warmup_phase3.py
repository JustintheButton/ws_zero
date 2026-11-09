"""Diagnostic: with the protocol's linear warmup, does the value head survive LR 0.05 and 0.2? And is the step-100
'shut off' flag real (training mode) or an artifact of the running statistics used in play mode?
    python scripts/warmup_phase3.py <lr:warmup:seed,...> <steps>"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import optax  # noqa: E402

import check_value_head as C  # noqa: E402
import ws_zero.net as N  # noqa: E402
from ws_zero.seeding import derive_seed  # noqa: E402
from ws_zero.train import make_optimizer, make_train_step  # noqa: E402

runs, steps = [r.split(":") for r in sys.argv[1].split(",")], int(sys.argv[2])
obs_all, masks_all = C.collect(16384 + 2048, derive_seed(0, "value-check-data"))
test_obs, test_mask = jnp.asarray(obs_all[16384:], dtype=jnp.float32), jnp.asarray(masks_all[16384:])
test_pi, test_z = C.targets(test_obs, test_mask)
ent = -jnp.mean(jnp.sum(jnp.where(test_pi > 0, test_pi * jnp.log(jnp.where(test_pi > 0, test_pi, 1.0)), 0.0), -1))


@jax.jit
def evaluate(params):
    logits, v, _, (vmap_play, hidden) = N._forward(params, test_obs, train=False)
    _, v_train, _, (vmap_train, _) = N._forward(params, test_obs, train=True)
    kl = -jnp.mean(jnp.sum(test_pi * N.masked_log_softmax(logits, test_mask), -1)) - ent
    return (jnp.mean((test_z - v) ** 2), kl, jnp.mean(vmap_play > 0), jnp.mean(vmap_train > 0),
            jnp.mean(jnp.abs(v) > 0.99), jnp.mean(jnp.abs(v_train) > 0.99), jnp.mean((test_z - v_train) ** 2))


for lr, warmup, seed in runs:
    lr, warmup, seed = float(lr), int(warmup), int(seed)
    schedule = optax.linear_schedule(0.0, lr, warmup) if warmup else lr
    opt = make_optimizer(schedule)
    params = N.init_params(jax.random.key(derive_seed(seed, "init")))
    state, step = opt.init(params), make_train_step(opt)
    rng = np.random.default_rng(derive_seed(seed, "batches"))
    log = []
    for i in range(steps + 1):
        if i % 25 == 0:
            log.append((i, *[float(x) for x in evaluate(params)]))
        if i < steps:
            idx = rng.integers(0, 16384, 1024)
            o, m = jnp.asarray(obs_all[idx], dtype=jnp.float32), jnp.asarray(masks_all[idx])
            p, z = C.targets(o, m)
            params, state, _, _ = step(params, state, (o, m, p, z))
    first = lambda k: next((r[0] for r in log if r[k] <= 0.01), None)  # noqa: E731
    sat = lambda k: next((r[0] for r in log if r[k] >= 0.5), None)  # noqa: E731
    e = log[-1]
    print(f"RESULT lr {lr} warmup {warmup} seed {seed}: value map off (play mode) first at {first(3)}, "
          f"(training mode) first at {first(4)}; saturated (play) first at {sat(5)}, (training) at {sat(6)} | "
          f"final value error play {e[1]:.3f} training {e[7]:.3f} | move KL {e[2]:.2f} | value map active at end "
          f"play {e[3]:.2f} training {e[4]:.2f}", flush=True)
    print("TRACE", lr, warmup, seed, " ".join(f"{r[0]}:{r[3]:.2f}/{r[4]:.2f}" for r in log if r[0] <= 400), flush=True)
