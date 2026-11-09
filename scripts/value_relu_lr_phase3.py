"""Diagnostic: does the value map's ReLU die at the larger D1 learning rates, and does removing it (keeping batch
norm) prevent that?  python scripts/value_relu_lr_phase3.py <current|linear> <lr:seed,lr:seed,...> <steps>"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

import check_value_head as C  # noqa: E402
import ws_zero.net as N  # noqa: E402
from ws_zero.seeding import derive_seed  # noqa: E402

variant, runs, steps = sys.argv[1], [r.split(":") for r in sys.argv[2].split(",")], int(sys.argv[3])

if variant == "linear":  # conv -> batch norm -> (no ReLU) -> fc1
    def _forward(params, obs, train):
        stats = {}

        def conv_bn(x, conv, bn):
            y, stats[f"{bn}/mean"], stats[f"{bn}/var"] = N.batch_norm(
                N._conv(x, params[f"{conv}/w"]), params[f"{bn}/scale"], params[f"{bn}/offset"],
                params[f"{bn}/mean"], params[f"{bn}/var"], train)
            return y

        x = jax.nn.relu(conv_bn(obs, "stem", "stem/bn"))
        for i in range(N.BLOCKS):
            h = jax.nn.relu(conv_bn(x, f"block{i}/conv1", f"block{i}/bn1"))
            x = jax.nn.relu(x + conv_bn(h, f"block{i}/conv2", f"block{i}/bn2"))
        p = jax.nn.relu(conv_bn(x, "policy/conv", "policy/bn"))
        logits = N.policy_to_actions(N._conv(p, params["policy/out/w"]) + params["policy/out/b"])
        value_map = conv_bn(x, "value/conv", "value/bn").reshape(x.shape[0], 64)
        hidden = value_map @ params["value/fc1/w"] + params["value/fc1/b"]
        v = jnp.tanh(jax.nn.relu(hidden) @ params["value/fc2/w"] + params["value/fc2/b"])[:, 0]
        return logits, v, stats, (value_map, hidden)

    N._forward = _forward

obs, masks = C.collect(16384 + 2048, derive_seed(0, "value-check-data"))
test = (jnp.asarray(obs[16384:], dtype=jnp.float32), jnp.asarray(masks[16384:]))
for lr, seed in runs:
    r = C.run_seed(int(seed), obs[:16384], masks[:16384], test, float(lr), steps, 1024)
    c = r["curve"]
    dead = next((x["step"] for x in c if x["conv_active"] <= 0.01 or x["fc1_active"] <= 0.01 or x["saturated"] >= 0.5), None)
    print(f"RESULT {variant} lr {lr} seed {seed}: {'PASS' if r['passed'] else 'FAIL'} | broken from step {dead} | value error "
          f"{c[0]['value_mse']:.3f} -> {c[-1]['value_mse']:.3f} (best {min(x['value_mse'] for x in c):.3f}) | move KL "
          f"{c[-1]['policy_kl']:.2f} | max saturated {max(x['saturated'] for x in c):.2f} | min value-layer active "
          f"{min(x['fc1_active'] for x in c):.2f}", flush=True)
