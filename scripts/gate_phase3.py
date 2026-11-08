"""Phase 3 gate: the network can overfit a small batch of positions.

256 positions from random pgx games, trained on as one full batch with the protocol optimizer (SGD, momentum 0.9,
weight decay 1e-4 on weights) at LR 0.003, the smallest in the D1 grid.

Targets (--targets):
  realistic (default): as in real training, the move target is a spread over the legal moves, not one move
    (softmax of 2 x the value of the piece captured), and the result target is who is ahead in material,
    tanh(material difference / 4) (scripts/check_value_head.py).
    Pass: move KL divergence < 0.05, value error <= 0.01, and a healthy value head at the end (fewer than half
    the values saturated, value map and value layer each active on more than 1% of units).
  random: one random legal move and a random result in {-1, 0, +1}, which can only be memorized (the first
    version of this gate; harsher than real training). Pass: every move target ranked first, value error <= 0.01.

    uv run python scripts/gate_phase3.py --seed 0 --out docs/gates/phase-3-overfit-seed0.json
"""
import argparse
import json
import time

import jax
import jax.numpy as jnp
import numpy as np

from check_value_head import targets as realistic_targets
from ws_zero.game.env import ChessEnv
from ws_zero.net import apply, init_params, masked_log_softmax, value_head_health
from ws_zero.seeding import derive_seed
from ws_zero.train import make_optimizer, make_train_step


def random_positions(n: int, seed: int, games: int = 64, plies: int = 160):
    """n distinct non-terminal positions from random games, with random one-hot move and random result targets."""
    env, rng = ChessEnv(), np.random.default_rng(seed)
    state = env.init(games)
    obs, masks = [], []
    for _ in range(plies):
        live = ~np.asarray(env.is_terminal(state))
        mask = np.asarray(env.legal_action_mask(state))
        obs.append(np.asarray(env.observe(state))[live]); masks.append(mask[live])
        noise = rng.random(mask.shape)
        actions = np.argmax(np.where(mask, noise, -1.0), axis=1)
        state = env.step(state, actions)
    obs, masks = np.concatenate(obs), np.concatenate(masks)
    _, unique = np.unique(obs.reshape(len(obs), -1), axis=0, return_index=True)
    pick = rng.choice(unique, size=n, replace=False)
    obs, masks = obs[pick], masks[pick]
    targets = np.array([rng.choice(np.flatnonzero(m)) for m in masks])
    pi = np.zeros(masks.shape, dtype=np.float32)
    pi[np.arange(n), targets] = 1.0
    z = rng.choice(np.array([-1.0, 0.0, 1.0], dtype=np.float32), size=n)
    return tuple(jnp.asarray(a) for a in (obs, masks, pi, z)), targets


def main():
    parser = argparse.ArgumentParser(description="Phase 3 gate: overfit a small batch")
    parser.add_argument("--targets", choices=["realistic", "random"], default="realistic")
    parser.add_argument("--positions", type=int, default=256)
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--lr", type=float, default=0.003)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    batch, top_targets = random_positions(args.positions, derive_seed(args.seed, "positions"))
    obs, mask, pi, z = batch
    if args.targets == "realistic":
        pi, z = realistic_targets(obs, mask)
        batch = (obs, mask, pi, z)
    entropy = -jnp.mean(jnp.sum(jnp.where(pi > 0, pi * jnp.log(jnp.where(pi > 0, pi, 1.0)), 0.0), -1))
    params = init_params(jax.random.key(derive_seed(args.seed, "init")))
    opt = make_optimizer(args.lr)
    opt_state, step = opt.init(params), make_train_step(opt)

    @jax.jit
    def metrics(params):
        logits, v = apply(params, obs)
        logp = masked_log_softmax(logits, mask)
        top1 = jnp.mean(jnp.argmax(logp, axis=-1) == jnp.asarray(top_targets))
        kl = -jnp.mean(jnp.sum(pi * logp, -1)) - entropy
        return top1, kl, jnp.mean((z - v) ** 2), value_head_health(params, obs)

    def passed(m):
        if args.targets == "random":
            return m["top1"] == 1.0 and m["value_mse"] <= 0.01
        return (m["policy_kl"] < 0.05 and m["value_mse"] <= 0.01 and m["saturated"] < 0.5
                and m["fc1_active"] > 0.01 and m["conv_active"] > 0.01)

    curve, passed_at, start = [], None, time.time()
    for i in range(args.steps + 1):
        if i % 25 == 0:
            top1, kl, mse, health = metrics(params)
            m = {"step": i, "top1": float(top1), "policy_kl": float(kl), "value_mse": float(mse),
                 **{k: float(v) for k, v in health.items()}}
            curve.append(m)
            if passed_at is None and passed(m):
                passed_at = i
            print(f"step {i:5d}  move KL {m['policy_kl']:.3f}  top-1 {m['top1']:6.1%}  value error {mse:.4f}  "
                  f"saturated {m['saturated']:.2f}  value map / layer active {m['conv_active']:.2f} / "
                  f"{m['fc1_active']:.2f}", flush=True)
            if passed_at is not None and i >= passed_at + 100:
                break
        if i < args.steps:
            params, opt_state, loss, parts = step(params, opt_state, batch)

    summary = {"targets": args.targets, "positions": args.positions, "lr": args.lr, "seed": args.seed,
               "passed": passed_at is not None, "passed_at_step": passed_at,
               "seconds": round(time.time() - start, 1), "device": str(jax.devices()[0]),
               "legal_moves_mean": float(np.asarray(mask).sum(1).mean()), "curve": curve}
    print(json.dumps({k: v for k, v in summary.items() if k != "curve"}, indent=2))
    if args.out:
        with open(args.out, "w") as f:
            json.dump(summary, f, indent=1)


if __name__ == "__main__":
    main()
