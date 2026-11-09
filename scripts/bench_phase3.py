"""Phase 3 throughput benchmark, and an estimate of self-play games per hour.

Measures, at several batch sizes (one batch = many games in parallel):
  - network evaluations per second (forward pass only),
  - pgx steps per second (step + observation),
  - one search simulation: pgx step + observation + network evaluation, which is the work of one MCTS simulation,
  - training steps per second at the protocol's batch of 1,024.

Estimate (protocol D2: chunks of 100 games played together, then training): a chunk's search lasts as long as
its longest game, at SIMULATIONS simulations per ply, with the batch equal to the number of games in flight; training
draws each position 4 times (positions x 4 / 1,024 steps). Batch 100 is one run per GPU; larger batches show what
several runs sharing one GPU could reach. Search bookkeeping is not included; Phase 4 measures real self-play.

    uv run python scripts/bench_phase3.py --out docs/gates/phase-3-bench-mac.json
"""
import argparse
import json
import platform
import time
from importlib.metadata import version

import jax
import jax.numpy as jnp
import numpy as np

from ws_zero.game.env import ChessEnv
from ws_zero.net import apply, init_params
from ws_zero.train import make_optimizer, make_train_step

SIMULATIONS = 800  # protocol §3 (200 until 2026-11-08)
TRAIN_BATCH = 1024
SAMPLES_PER_POSITION = 4
AVERAGE_PLIES = 340  # average length of the Phase 2 random games
LONGEST_PLIES = (512, 340)  # chunk length: conservative (some game reaches the 512-ply cap) and optimistic


def median_seconds(fn, *args, repeats: int) -> float:
    jax.block_until_ready(fn(*args))  # compile and warm up
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        jax.block_until_ready(fn(*args))
        times.append(time.perf_counter() - start)
    return float(np.median(times))


def main():
    parser = argparse.ArgumentParser(description="Phase 3 throughput benchmark")
    parser.add_argument("--batches", default="100,400,1600,6400")
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--label", default=platform.node())
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    batches = [int(b) for b in args.batches.split(",")]

    env = ChessEnv()
    params = init_params(jax.random.key(0))
    forward = jax.jit(apply)

    def random_actions(state, key):
        noise = jax.random.uniform(key, state.legal_action_mask.shape)
        return jnp.argmax(jnp.where(state.legal_action_mask, noise, -1.0), axis=-1)

    @jax.jit
    def env_step(state, key):
        state = env._step(state, random_actions(state, key))
        return state, env._observe(state)

    @jax.jit
    def simulation(params, state, key):
        state = env._step(state, random_actions(state, key))
        return state, apply(params, env._observe(state))

    def midgame(batch):  # 20 random plies in, so positions are not all the start position
        state, key = env.init(batch), jax.random.key(1)
        for _ in range(20):
            key, sub = jax.random.split(key)
            state, _ = env_step(state, sub)
        return state

    rows = []
    for b in batches:
        state = midgame(b)
        obs = env.observe(state)
        key = jax.random.key(2)
        t_fwd = median_seconds(forward, params, obs, repeats=args.repeats)
        t_env = median_seconds(env_step, state, key, repeats=args.repeats)
        t_sim = median_seconds(simulation, params, state, key, repeats=args.repeats)
        rows.append({"batch": b, "forward_s": t_fwd, "env_s": t_env, "simulation_s": t_sim,
                     "evals_per_s": b / t_fwd, "env_steps_per_s": b / t_env, "simulations_per_s": b / t_sim})
        print(f"batch {b:6d}: network {b / t_fwd:10,.0f} evals/s | pgx {b / t_env:10,.0f} steps/s | "
              f"step + eval {b / t_sim:10,.0f} /s", flush=True)

    state = midgame(TRAIN_BATCH)
    mask = state.legal_action_mask
    batch = (env.observe(state), mask, mask / mask.sum(-1, keepdims=True), jnp.zeros(TRAIN_BATCH))
    opt = make_optimizer(0.01)
    step = make_train_step(opt)
    opt_state = opt.init(params)
    t_train = median_seconds(lambda p, s: step(p, s, batch)[2], params, opt_state, repeats=args.repeats)
    print(f"training: {1 / t_train:,.1f} steps/s at batch {TRAIN_BATCH} ({TRAIN_BATCH / t_train:,.0f} positions/s)")

    estimates = []
    for r in rows:
        train_s = r["batch"] * AVERAGE_PLIES * SAMPLES_PER_POSITION / TRAIN_BATCH * t_train
        for longest in LONGEST_PLIES:
            search_s = longest * SIMULATIONS * r["simulation_s"]
            estimates.append({"batch": r["batch"], "chunk_plies": longest, "average_plies": AVERAGE_PLIES,
                              "games_per_hour": 3600 * r["batch"] / (search_s + train_s),
                              "training_share": train_s / (search_s + train_s)})
        print(f"estimate, {r['batch']} games in flight: "
              + ", ".join(f"{e['games_per_hour']:,.0f} games/hour (chunk {e['chunk_plies']} plies)"
                          for e in estimates[-2:]))

    summary = {"label": args.label, "device": str(jax.devices()[0]), "device_kind": jax.devices()[0].device_kind,
               "platform": platform.platform(), "jax": jax.__version__, "pgx": version("pgx"),
               "simulations_per_move": SIMULATIONS, "rows": rows,
               "train": {"batch": TRAIN_BATCH, "step_s": t_train, "positions_per_s": TRAIN_BATCH / t_train},
               "estimates": estimates, "note": "estimates exclude search bookkeeping (measured in Phase 4)"}
    if args.out:
        with open(args.out, "w") as f:
            json.dump(summary, f, indent=1)


if __name__ == "__main__":
    main()
