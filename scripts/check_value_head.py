"""Phase 3 follow-up: does the value head stay alive under realistic training conditions?

The memorization gate asks for random labels, which is harsher than real training. Here the targets are learnable
functions of the position, soft, and drawn in minibatches from many positions, as in real training:
  - result target: who is ahead in material, z = tanh(material difference / 4), from the side to move's view
    (pawn 1, knight 3, bishop 3, rook 5, queen 9);
  - move target: a spread over the legal moves, softmax(2 x value of the piece captured), so captures of big pieces
    are favoured and quiet moves share the rest.
The same random starts as the gate (seed k = init key derive_seed(k, "init")). Protocol optimizer, batch 1,024.

Pass: in every seed the value head stays healthy at every check (value map and value layer each active on more than
1% of units, fewer than half the values saturated) and the held-out value error falls below half its starting value.

History (protocol §10): without batch norm, the original value head (ReLU after the 1x1 value conv) died in all
three seeds within 300 steps; with that ReLU removed it learned over 900 steps, but over 3,000 steps one seed in
three saturated for good. Batch norm was then restored, with AlphaZero's original value head.

    uv run python scripts/check_value_head.py --seeds 0,1,2 --out docs/gates/phase-3-value-check.json
"""
import argparse
import json
import time

import jax
import jax.numpy as jnp
import numpy as np

from ws_zero.game.env import ChessEnv
from ws_zero.game.moves import _DELTAS, NUM_ACTIONS
from ws_zero.net import apply, init_params, masked_log_softmax, value_head_health
from ws_zero.seeding import derive_seed
from ws_zero.train import make_optimizer, make_train_step

PIECE_VALUES = jnp.array([1.0, 3.0, 3.0, 5.0, 9.0, 0.0])  # pawn, knight, bishop, rook, queen, king


def _target_squares() -> np.ndarray:
    """For each of the 4,672 actions, the destination as an observation index row * 8 + column (64 if off-board)."""
    to = np.full(NUM_ACTIONS, 64, dtype=np.int32)
    for a in range(NUM_ACTIONS):
        f, plane = divmod(a, 73)
        col, rank = divmod(f, 8)
        d_rank, d_file = (1, (0, 1, -1)[plane % 3]) if plane < 9 else _DELTAS[plane - 9]
        r, c = rank + d_rank, col + d_file
        if 0 <= r < 8 and 0 <= c < 8:
            to[a] = (7 - r) * 8 + c
    return to


TARGET_SQUARE = jnp.asarray(_target_squares())


@jax.jit
def targets(obs, mask):
    mine = jnp.einsum("bhwp,p->b", obs[..., 0:6], PIECE_VALUES)
    theirs_map = jnp.einsum("bhwp,p->bhw", obs[..., 6:12], PIECE_VALUES).reshape(obs.shape[0], 64)
    z = jnp.tanh((mine - theirs_map.sum(-1)) / 4.0)
    captured = jnp.concatenate([theirs_map, jnp.zeros((obs.shape[0], 1))], axis=1)[:, TARGET_SQUARE]
    pi = jax.nn.softmax(jnp.where(mask, 2.0 * captured, -1e9), axis=-1)
    return pi, z


def collect(n: int, seed: int, games: int = 256, keep: float = 0.5):
    """n positions from random games (plies 4 and later), stored as float16 observations and legal-move masks."""
    env, rng = ChessEnv(), np.random.default_rng(seed)
    state, obs, masks, total, ply = env.init(games), [], [], 0, 0
    while total < n:
        live = ~np.asarray(env.is_terminal(state))
        mask = np.asarray(env.legal_action_mask(state))
        if ply >= 4:
            take = live & (rng.random(games) < keep)
            obs.append(np.asarray(env.observe(state))[take].astype(np.float16)); masks.append(mask[take])
            total += int(take.sum())
        if not live.any():
            state, ply = env.init(games), 0
            continue
        state = env.step(state, np.argmax(np.where(mask, rng.random(mask.shape), -1.0), axis=1))
        ply += 1
    obs, masks = np.concatenate(obs)[:n], np.concatenate(masks)[:n]
    order = rng.permutation(n)
    return obs[order], masks[order]


def run_seed(seed, train_obs, train_mask, test, lr, steps, batch):
    params = init_params(jax.random.key(derive_seed(seed, "init")))
    opt = make_optimizer(lr)
    opt_state, step = opt.init(params), make_train_step(opt)
    test_obs, test_mask = test
    test_pi, test_z = targets(test_obs, test_mask)
    test_entropy = -jnp.mean(jnp.sum(jnp.where(test_pi > 0, test_pi * jnp.log(test_pi), 0.0), -1))

    @jax.jit
    def evaluate(params):
        logits, v = apply(params, test_obs)
        kl = -jnp.mean(jnp.sum(test_pi * masked_log_softmax(logits, test_mask), -1)) - test_entropy
        return jnp.mean((test_z - v) ** 2), kl, value_head_health(params, test_obs)

    rng, curve = np.random.default_rng(derive_seed(seed, "batches")), []
    for i in range(steps + 1):
        if i % 100 == 0:
            mse, kl, health = evaluate(params)
            c = {"step": i, "value_mse": float(mse), "policy_kl": float(kl),
                 **{k: float(v) for k, v in health.items()}}
            curve.append(c)
            print(f"seed {seed} step {i:4d}  held-out value error {c['value_mse']:.4f}  move KL {c['policy_kl']:.3f}  "
                  f"value map / layer active {c['conv_active']:.2f} / {c['fc1_active']:.2f}  saturated {c['saturated']:.2f}",
                  flush=True)
        if i < steps:
            idx = rng.integers(0, len(train_obs), batch)
            obs = jnp.asarray(train_obs[idx], dtype=jnp.float32)
            mask = jnp.asarray(train_mask[idx])
            pi, z = targets(obs, mask)
            params, opt_state, _, _ = step(params, opt_state, (obs, mask, pi, z))
    alive = all(c["conv_active"] > 0.01 and c["fc1_active"] > 0.01 and c["saturated"] < 0.5 for c in curve)
    learned = curve[-1]["value_mse"] < 0.5 * curve[0]["value_mse"]
    return {"seed": seed, "passed": alive and learned, "value_head_alive": alive, "value_learned": learned,
            "curve": curve}


def main():
    parser = argparse.ArgumentParser(description="Phase 3 follow-up: value head under realistic training")
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--lr", type=float, default=0.003)
    parser.add_argument("--steps", type=int, default=900)
    parser.add_argument("--batch", type=int, default=1024)
    parser.add_argument("--train", type=int, default=16384)
    parser.add_argument("--test", type=int, default=2048)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    start = time.time()
    obs, masks = collect(args.train + args.test, derive_seed(0, "value-check-data"))
    test = (jnp.asarray(obs[args.train:], dtype=jnp.float32), jnp.asarray(masks[args.train:]))
    results = [run_seed(int(s), obs[:args.train], masks[:args.train], test, args.lr, args.steps, args.batch)
               for s in args.seeds.split(",")]
    for r in results:
        print(f"seed {r['seed']}: {'PASS' if r['passed'] else 'FAIL'} (value head healthy throughout: "
              f"{r['value_head_alive']}, value error halved: {r['value_learned']})")
    summary = {"lr": args.lr, "steps": args.steps, "batch": args.batch, "train_positions": args.train,
               "test_positions": args.test, "seconds": round(time.time() - start, 1), "results": results}
    if args.out:
        with open(args.out, "w") as f:
            json.dump(summary, f, indent=1)


if __name__ == "__main__":
    main()
