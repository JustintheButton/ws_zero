"""Demo games for the arena's archive and score tables, played by the stand-in models before any training.

Real games played with the real search (ws_zero.search), in the record format the training and evaluation runs
will use: for every ply the move (UCI and SAN), and for a searching player its root value (for the side to move)
and its three most-visited moves. Writes arena/data/demo-games.json.

- Self-play: the untrained network against itself, as in self-play (root noise, moves sampled for 30 plies).
- Testing: each stand-in against a random mover, half the games with each color, as in evaluation (no noise,
  most-visited move).

    uv run python scripts/arena_demo_games.py --simulations 50 --games 10
"""
import argparse
import json
import math
import time
from pathlib import Path

import chess
import jax
import jax.numpy as jnp
import numpy as np

from ws_zero.arena import pawn_pusher, untrained
from ws_zero.game.env import ChessEnv
from ws_zero.game.moves import NUM_ACTIONS, action_to_move
from ws_zero.net import apply
from ws_zero.search import search_moves

ROOT = Path(__file__).resolve().parents[1]
PIECE_VALUES = jnp.array([1.0, 3.0, 3.0, 5.0, 9.0, 0.0])


def material_counter(params, obs):
    mine = jnp.einsum("bhwp,p->b", obs[..., 0:6], PIECE_VALUES)
    theirs = jnp.einsum("bhwp,p->b", obs[..., 6:12], PIECE_VALUES)
    return jnp.zeros((obs.shape[0], NUM_ACTIONS)), jnp.tanh((mine - theirs) / 4)


PLAYERS = {  # id -> (name, evaluate, params)
    "untrained": ("Untrained network", apply, untrained()),
    "pawn-pusher": ("Hand-written pawn-pusher", apply, pawn_pusher()),
    "material": ("Material counter", material_counter, None),
}


def play(env, player, sides, simulations, self_play, seed):
    """Play len(sides) games together. sides[i]: "both" (self-play), "w" or "b" (the player's color; a random
    mover has the other). Returns one record per game."""
    _, evaluate, params = PLAYERS[player]
    n = len(sides)
    search = jax.jit(lambda state, key: search_moves(env, evaluate, params, key, state, simulations, self_play))
    rng = np.random.default_rng(seed)
    state, key = env.init(n), jax.random.key(seed)
    boards = [chess.Board() for _ in range(n)]
    records = [{"moves": [], "san": [], "values": [], "top": []} for _ in range(n)]
    done = np.zeros(n, bool)
    while not done.all():
        key, sub = jax.random.split(key)
        actions, out = search(state, sub)
        summary = out.search_tree.summary()
        visits, root_values = np.asarray(summary.visit_counts), np.asarray(summary.value)
        actions, legal = np.asarray(actions), np.asarray(state.legal_action_mask)
        chosen = actions.copy()
        for i in range(n):
            if done[i]:
                continue
            board, rec = boards[i], records[i]
            mover = "w" if board.turn == chess.WHITE else "b"
            searching = sides[i] == "both" or sides[i] == mover
            if not searching:
                chosen[i] = rng.choice(np.flatnonzero(legal[i]))
            move = action_to_move(board, int(chosen[i]))
            if searching:
                top = [a for a in np.argsort(-visits[i], kind="stable")[:3] if visits[i, a] > 0]
                rec["values"].append(round(float(root_values[i]), 3))
                rec["top"].append([[board.san(action_to_move(board, int(a))), int(visits[i, a])] for a in top])
            else:
                rec["values"].append(None)
                rec["top"].append(None)
            rec["san"].append(board.san(move))
            rec["moves"].append(move.uci())
            board.push(move)
        state = env.step(state, jnp.asarray(chosen))
        over = np.asarray(env.is_terminal(state))
        value = np.asarray(env.terminal_value(state))
        for i in np.flatnonzero(over & ~done):
            done[i] = True
            board, rec = boards[i], records[i]
            if value[i] == -1:  # the side to move is checkmated
                rec["result"] = "0-1" if board.turn == chess.WHITE else "1-0"
                rec["end"] = "checkmate"
            else:
                rec["result"] = "1/2-1/2"
                rec["end"] = ("stalemate" if board.is_stalemate() else "insufficient material"
                              if board.is_insufficient_material() else "fifty-move rule" if board.halfmove_clock >= 100
                              else "512-ply cap" if board.ply() >= 512 else "threefold repetition")
    return records


def score_cell(records, sides, ids):
    """The player's score against the random mover, with a 95% Wilson interval."""
    points = 0.0
    for rec, side in zip(records, sides):
        points += {"1/2-1/2": 0.5}.get(rec["result"], 1.0 if (rec["result"] == "1-0") == (side == "w") else 0.0)
    n, p, z = len(records), points / len(records), 1.96
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return {"score": round(p, 3), "lo": round(max(0.0, centre - half), 3), "hi": round(min(1.0, centre + half), 3),
            "n": n, "games": ids}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--simulations", type=int, default=50)
    parser.add_argument("--games", type=int, default=10)
    parser.add_argument("--out", default=str(ROOT / "arena" / "data" / "demo-games.json"))
    args = parser.parse_args()
    env, games, start = ChessEnv(), {}, time.time()

    sides = ["both"] * args.games
    records = play(env, "untrained", sides, args.simulations, self_play=True, seed=0)
    selfplay_ids = []
    for k, rec in enumerate(records):
        gid = f"demo-selfplay-{k + 1}"
        games[gid] = {**rec, "white": "Untrained network", "black": "Untrained network", "kind": "self-play"}
        selfplay_ids.append(gid)
    print(f"self-play: {len(records)} games, {time.time() - start:.0f} s", flush=True)

    rows = []
    for pid in PLAYERS:
        name = PLAYERS[pid][0]
        sides = ["w" if k % 2 == 0 else "b" for k in range(args.games)]
        records = play(env, pid, sides, args.simulations, self_play=False, seed=1)
        ids = []
        for k, (rec, side) in enumerate(zip(records, sides)):
            gid = f"demo-{pid}-vs-random-{k + 1}"
            games[gid] = {**rec, "white": name if side == "w" else "Random mover",
                          "black": name if side == "b" else "Random mover", "kind": "testing"}
            ids.append(gid)
        rows.append({"label": name, "model": pid, "cells": [score_cell(records, sides, ids)]})
        print(f"{name} vs random mover: {rows[-1]['cells'][0]['score']:.2f}, {time.time() - start:.0f} s", flush=True)

    data = {
        "demo": True, "simulations": args.simulations,
        "runs": [{"id": "demo-untrained", "label": "Stand-in: untrained network (demo)", "arm": "stand-in",
                  "checkpoints": [{"games": 0, "model": "untrained", "selfplay": selfplay_ids}]}],
        "tables": [{"id": "demo-ladder", "title": f"Stand-ins against a random mover (demo, {args.simulations} "
                    "look-aheads)", "opponents": ["Random mover"], "rows": rows}],
        "games": games,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(data, separators=(",", ":")))
    print(f"wrote {args.out} ({len(games)} games, {time.time() - start:.0f} s)")


if __name__ == "__main__":
    main()
