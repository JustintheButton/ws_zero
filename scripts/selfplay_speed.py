"""Phase 4: real self-play speed, search included, and the games-per-hour estimate it gives.

Plays games from the start position as self-play does (800 simulations, root noise, moves sampled for the first 30
plies), with an untrained network. That is the cheapest case: an untrained network spreads its look-aheads evenly, so
its search trees are only a few moves deep, and every look-ahead in the batch walks down to the deepest line of any game.
A trained network makes deeper trees and slower plies (on a Modal L4, 100 games x 800 simulations: 0.86 s per ply
with the deepest line 4 moves, 1.08 s at 8, 1.54 s at 19, 1.87 s at 25; docs/gates/phase-4.md).
Times every ply after the first (which compiles), and one training step. The estimate follows the Phase 3 benchmark (protocol D2): games are played
in chunks, a chunk lasts as long as its longest game (512 plies if one reaches the cap, 340 if none is longer than
the average random game), and then training draws each new position 4 times in batches of 1,024.

--prior-peak makes deeper trees without a trained network: the network still runs, but move a's logit is set to
-peak x a, so the look-aheads concentrate on a few moves, as a trained network's do. The script reports how deep the
trees were, so that a time can later be matched to a trained network's real depth.

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


def tree_depths(parents: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """For each game, the deepest line in its search tree and the mean depth of its nodes, in moves below the root.
    `parents` is mctx's table (games, nodes): node 0 is the root, a node's parent has a smaller index, and -1 marks
    the root and any node never created."""
    games = np.arange(parents.shape[0])
    depth = np.zeros(parents.shape, dtype=int)
    for i in range(1, parents.shape[1]):
        depth[:, i] = np.where(parents[:, i] >= 0, depth[games, np.maximum(parents[:, i], 0)] + 1, 0)
    created = parents >= 0
    return depth.max(axis=1), depth.sum(axis=1) / np.maximum(created.sum(axis=1), 1)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--games", type=int, default=100, help="games played together (protocol D2: 100)")
    parser.add_argument("--plies", type=int, default=40, help="plies to time, after the compiling one")
    parser.add_argument("--simulations", type=int, default=800)
    parser.add_argument("--prior-peak", type=float, default=0.0,
                        help="0: the untrained network's own move probabilities (shallow trees); above 0: logit of "
                             "move a = -peak x a (deeper trees)")
    parser.add_argument("--label", default=platform.node())
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    env = ChessEnv()
    params = init_params(jax.random.key(0))

    def evaluate(params, obs):
        logits, value = apply(params, obs)
        if args.prior_peak:  # 0 x logits keeps the whole network in the compiled program
            logits = 0.0 * logits - args.prior_peak * jnp.arange(logits.shape[-1], dtype=logits.dtype)
        return logits, value

    @jax.jit
    def ply(params, state, key):
        actions, out = search_moves(env, evaluate, params, key, state, args.simulations, self_play=True)
        return env.step(state, actions), out.search_tree.parents

    state = env.init(args.games)
    keys = jax.random.split(jax.random.key(1), args.plies + 1)
    start = time.perf_counter()
    compiled = ply.lower(params, state, keys[0]).compile()
    compile_s = time.perf_counter() - start
    print(f"compiled in {compile_s:.1f} s", flush=True)
    ply_s, deepest, mean_depth = [], [], []
    for i, key in enumerate(keys):
        start = time.perf_counter()
        state, parents = jax.block_until_ready(compiled(params, state, key))
        elapsed = time.perf_counter() - start
        deep, mean = tree_depths(np.asarray(parents))
        print(f"ply {i + 1}: {elapsed:.3f} s, deepest line {deep.max()} moves, mean node depth {mean.mean():.1f}",
              flush=True)
        if i:  # the first run after compiling also loads the program onto the device
            ply_s.append(elapsed)
            deepest.append(int(deep.max()))  # every look-ahead in the batch waits for the deepest line of any game
            mean_depth.append(float(mean.mean()))
    per_ply = float(np.median(ply_s))
    finished = int(np.asarray(env.is_terminal(state)).sum())
    print(f"{args.games} games, {args.simulations} simulations: compiled in {compile_s:.1f} s, then "
          f"median {per_ply:.3f} s per ply (min {min(ply_s):.3f}, max {max(ply_s):.3f}) over {len(ply_s)} plies; "
          f"{finished} games finished; deepest line (median over plies) {int(np.median(deepest))} moves, "
          f"mean node depth {np.median(mean_depth):.1f}", flush=True)

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
               "games": args.games, "simulations": args.simulations, "prior_peak": args.prior_peak,
               "compile_s": compile_s, "ply_s": ply_s, "deepest_line": deepest, "mean_node_depth": mean_depth,
               "median_ply_s": per_ply, "finished_games": finished, "train_step_s": train_step_s,
               "estimates": estimates}
    if args.out:
        with open(args.out, "w") as f:
            json.dump(summary, f, indent=1)


if __name__ == "__main__":
    main()
