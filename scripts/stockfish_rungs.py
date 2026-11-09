"""Phase 5: what Stockfish really does at each rung of the node ladder (protocol §8.2).

For 200 positions from random games (6 to 60 random plies from the start, seed 0), at each node count: how many
nodes the engine reports having searched, how deep it got, and how often it plays the same move as at the next rung
up. Each position is searched as a new game.

    uv run python scripts/stockfish_rungs.py --out docs/gates/phase-5-stockfish-rungs.json
"""
import argparse
import json
import random
import statistics

import chess
import chess.engine

from ws_zero.stockfish import Stockfish

RUNGS = [1, 3, 10, 30, 100, 300, 1000, 3000, 10000]  # protocol §8.2


def positions(count: int, seed: int) -> list[str]:
    rng, fens = random.Random(seed), []
    while len(fens) < count:
        board = chess.Board()
        for _ in range(rng.randint(6, 60)):
            if board.is_game_over():
                break
            board.push(rng.choice(list(board.legal_moves)))
        if not board.is_game_over():
            fens.append(board.fen())
    return fens


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--positions", type=int, default=200)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    fens = positions(args.positions, seed=0)
    rows = []
    for nodes in RUNGS:
        moves, searched, depths = [], [], []
        with Stockfish(nodes) as engine:
            for fen in fens:
                result = engine.engine.play(chess.Board(fen), engine.limit, game=object(),
                                            info=chess.engine.INFO_BASIC)
                moves.append(result.move.uci())
                searched.append(result.info.get("nodes", 0))
                depths.append(result.info.get("depth", 0))
        rows.append({"nodes": nodes, "moves": moves, "searched_median": statistics.median(searched),
                     "searched_max": max(searched), "depth_median": statistics.median(depths)})
    print(f"{'rung':>6}  {'nodes searched: median':>22}  {'max':>6}  {'depth':>5}  same move as the next rung up")
    for row, above in zip(rows, rows[1:] + [None]):
        row["same_as_next"] = None if above is None else sum(a == b for a, b in zip(row["moves"], above["moves"]))
        same = "" if above is None else f"{row['same_as_next']} of {len(fens)}"
        print(f"{row['nodes']:6d}  {row['searched_median']:22.0f}  {row['searched_max']:6d}  "
              f"{row['depth_median']:5.0f}  {same}")
    if args.out:
        with open(args.out, "w") as f:
            json.dump({"positions": fens, "rungs": rows}, f, indent=1)


if __name__ == "__main__":
    main()
