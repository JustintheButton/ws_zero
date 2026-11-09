"""Phase 4: real self-play speed, search included, and the games-per-hour estimate it gives.

Plays games from the start position as self-play does (800 simulations, root noise, moves sampled for the first 30
plies), with an untrained network: the time per ply does not depend on the weights. Times every ply after the first
(which compiles), and one training step. The estimate follows the Phase 3 benchmark (protocol D2): games are played
in chunks, a chunk lasts as long as its longest game (512 plies if one reaches the cap, 340 if none is longer than
the average random game), and then training draws each new position 4 times in batches of 1,024.

    uv run python scripts/selfplay_speed.py --games 4 --plies 3            # quick check that it runs
    uv run modal run scripts/modal_selfplay_speed.py                         # the measurement, on one Modal L4
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
from ws_zero.search import search_moves
from ws_zero.train import make_optimizer, make_train_step

TRAIN_BATCH = 1024
SAMPLES_PER_POSITION = 4
AVERAGE_PLIES = 340  # average length of the Phase 2 random games
LONGEST_PLIES = (512, 340)  # chunk length: some game reaches the 512-ply cap, or none is longer than average


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--games", type=int, default=100, help="games played together (protocol D2: 100)")
    parser.add_argument("--plies", type=int, default=40, help="plies to time, after the compiling one")
    parser.add_argument("--simulations", type=int, default=800)
    parser.add_argument("--label", default=platform.node())
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    env = ChessEnv()
    params = init_params(jax.random.key(0))

    @jax.jit
    def ply(params, state, key):
        actions, _ = search_moves(env, apply, params, key, state, args.simulations, self_play=True)
        return env.step(state, actions)

    state = env.init(args.games)
    keys = jax.random.split(jax.random.key(1), args.plies + 1)
    start = time.perf_counter()
    state = jax.block_until_ready(ply(params, state, keys[0]))
    compile_s = time.perf_counter() - start
    ply_s = []
    for key in keys[1:]:
        start = time.perf_counter()
        state = jax.block_until_ready(ply(params, state, key))
        ply_s.append(time.perf_counter() - start)
    per_ply = float(np.median(ply_s))
    finished = int(np.asarray(env.is_terminal(state)).sum())
    print(f"{args.games} games, {args.simulations} simulations: first ply (compiles) {compile_s:.1f} s, then "
          f"median {per_ply:.3f} s per ply (min {min(ply_s):.3f}, max {max(ply_s):.3f}) over {len(ply_s)} plies; "
          f"{finished} games finished", flush=True)

    train_state = env.init(TRAIN_BATCH)
    mask = train_state.legal_action_mask
    batch = (env.observe(train_state), mask, mask / mask.sum(-1, keepdims=True), jnp.zeros(TRAIN_BATCH))
    step = make_train_step(make_optimizer(0.01))
    opt_state = make_optimizer(0.01).init(params)
    jax.block_until_ready(step(params, opt_state, batch)[2])
    times = []
    for _ in range(5):
        start = time.perf_counter()
        jax.block_until_ready(step(params, opt_state, batch)[2])
        times.append(time.perf_counter() - start)
    train_step_s = float(np.median(times))
    print(f"training: {train_step_s:.3f} s per step of {TRAIN_BATCH}", flush=True)

    train_s = args.games * AVERAGE_PLIES * SAMPLES_PER_POSITION / TRAIN_BATCH * train_step_s
    estimates = []
    for longest in LONGEST_PLIES:
        search_s = longest * per_ply
        estimates.append({"chunk_plies": longest, "games_per_hour": 3600 * args.games / (search_s + train_s),
                          "training_share": train_s / (search_s + train_s)})
        print(f"estimate, chunk of {longest} plies: {estimates[-1]['games_per_hour']:,.0f} games/hour", flush=True)

    summary = {"label": args.label, "device": str(jax.devices()[0]), "device_kind": jax.devices()[0].device_kind,
               "platform": platform.platform(), "jax": jax.__version__, "pgx": version("pgx"), "mctx": version("mctx"),
               "games": args.games, "simulations": args.simulations, "compile_s": compile_s, "ply_s": ply_s,
               "median_ply_s": per_ply, "finished_games": finished, "train_step_s": train_step_s,
               "estimates": estimates}
    if args.out:
        with open(args.out, "w") as f:
            json.dump(summary, f, indent=1)


if __name__ == "__main__":
    main()
