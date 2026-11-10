"""Make the opening suites (protocol §8.3): random 8-ply lines from the start position, kept only if the pinned
Stockfish, searching to depth 20, rates the position reached within 50 centipawns of equal.

Lines are drawn in a fixed order from the seed (each move uniformly at random among the legal moves); lines that end
the game or reach a position already seen are skipped. The first 50 kept are the checkpoint suite and the next 100
the GO / NO-GO suite. Stockfish at a fixed depth on one thread gives the same evaluation on any machine, so the
same seed gives the same suites anywhere.

    uv run modal run scripts/modal_stockfish_job.py --script make_openings --out data/openings.json
"""
import argparse
import itertools
import json
import random
from multiprocessing import Pool

import chess

from ws_zero.stockfish import NAME, Analyst

PLIES, DEPTH, LIMIT_CP = 8, 20, 50
SUITES = {"checkpoint": 50, "go_no_go": 100}


def candidate_lines(seed: int):
    """Random PLIES-ply lines (UCI) from the start position, without finished games or repeated final positions."""
    rng, seen = random.Random(seed), set()
    while True:
        board = chess.Board()
        for _ in range(PLIES):
            if board.is_game_over():
                break
            board.push(rng.choice(sorted(board.legal_moves, key=chess.Move.uci)))
        key = " ".join(board.fen().split()[:4])
        if len(board.move_stack) == PLIES and not board.is_game_over() and key not in seen:
            seen.add(key)
            yield [m.uci() for m in board.move_stack]


def select(scored, count: int, limit: int) -> list:
    """The first `count` (line, centipawns) pairs whose evaluation is within `limit` (a forced mate, None, is not)."""
    kept = []
    for line, centipawns in scored:
        if centipawns is not None and abs(centipawns) <= limit:
            kept.append((line, centipawns))
            if len(kept) == count:
                break
    return kept


_analyst = None


def _start(depth):
    global _analyst
    _analyst = Analyst(depth)


def _score(line):
    board = chess.Board()
    for uci in line:
        board.push_uci(uci)
    return _analyst.centipawns(board)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--depth", type=int, default=DEPTH)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    total, examined = sum(SUITES.values()), 0

    def scored(pool):
        nonlocal examined
        lines = candidate_lines(args.seed)
        while True:  # in order, a batch at a time, so the result does not depend on the number of workers
            batch = list(itertools.islice(lines, 8 * args.workers))
            for line, centipawns in zip(batch, pool.map(_score, batch)):
                examined += 1
                yield line, centipawns

    with Pool(args.workers, initializer=_start, initargs=(args.depth,)) as pool:
        kept = select(scored(pool), total, LIMIT_CP)
    entries = []
    for line, centipawns in kept:
        board = chess.Board()
        for uci in line:
            board.push_uci(uci)
        entries.append({"moves": line, "fen": board.fen(), "centipawns": centipawns})
    out = {"generator": {"seed": args.seed, "plies": PLIES, "engine": NAME, "depth": args.depth,
                         "limit_centipawns": LIMIT_CP, "examined": examined, "kept": len(entries)},
           "checkpoint": entries[:SUITES["checkpoint"]], "go_no_go": entries[SUITES["checkpoint"]:]}
    print(f"kept {len(entries)} of {examined} lines examined", flush=True)
    if args.out:
        with open(args.out, "w") as f:
            json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
