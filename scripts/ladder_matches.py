"""Phase 5: anchors of the evaluation ladder play each other (protocol §8.2), every pairing, both colours.

Until the protocol's opening suite exists (§8.3), the openings are random 8-ply sequences (seeded). Each pairing plays
every opening twice, colours reversed. Reports each pairing's score with a 95% range and the Elo difference it
implies; a full rating fit (Ordo) comes with the calibration proper.

    uv run python scripts/ladder_matches.py --players random,grabber,sf1320,sf1 --openings 10 --out build/ladder.json
"""
import argparse
import gzip
import itertools
import json
import math
import random
from dataclasses import asdict
from multiprocessing import Pool

import chess
import numpy as np

from ws_zero.players import Diluted, MaterialGrabber, RandomMover, play_game
from ws_zero.stockfish import DILUTED_RANDOM_SHARES, LADDER_NODES, LIMITED_NODES, MIN_ELO, Stockfish

PLAYERS = {"random": lambda: RandomMover(seed=0), "grabber": lambda: MaterialGrabber(seed=1),
           "sf1320": lambda: Stockfish(LIMITED_NODES, elo=MIN_ELO),
           **{f"sf{n}": (lambda n=n: Stockfish(n)) for n in LADDER_NODES},
           # the 1-node Stockfish playing a random move some share of the time: sf1r75 = 75% random moves
           **{f"sf1r{round(100 * share)}": (lambda share=share: Diluted(Stockfish(1), share, seed=2))
              for share in DILUTED_RANDOM_SHARES}}


def random_openings(count: int, seed: int, plies: int = 8) -> list[list[str]]:
    rng, out = random.Random(seed), []
    while len(out) < count:
        board = chess.Board()
        for _ in range(plies):
            board.push(rng.choice(sorted(board.legal_moves, key=chess.Move.uci)))
            if board.is_game_over():
                break
        else:
            out.append([m.uci() for m in board.move_stack])
    return out


def elo(score: float) -> float | None:
    return None if score <= 0 or score >= 1 else 400 * math.log10(score / (1 - score))


def fit_ratings(names: list[str], pairings: list[dict]) -> dict[str, float]:
    """One Elo rating per player from all pairings together (the Bradley-Terry model, a draw counting as half a win
    and half a loss), with the first player at 0. A stand-in for Ordo until the calibration proper."""
    index = {name: i for i, name in enumerate(names)}
    points, games = np.zeros((len(names), len(names))), np.zeros((len(names), len(names)))
    for p in pairings:
        low, high = index[p["low"]], index[p["high"]]
        points[high, low], points[low, high] = p["score"] * p["games"], (1 - p["score"]) * p["games"]
        games[high, low] = games[low, high] = p["games"]
    strength = np.ones(len(names))
    for _ in range(20000):
        strength = points.sum(axis=1) / (games / (strength[:, None] + strength[None, :])).sum(axis=1)
        strength /= strength[0]
    return {name: float(400 * np.log10(max(strength[i], 1e-300))) for name, i in index.items()}


def match(job):
    low, high, openings = job
    a, b = PLAYERS[low](), PLAYERS[high]()
    games, points = [], []
    for i, opening in enumerate(openings):
        for high_is_white in (True, False):
            white, black = (b, a) if high_is_white else (a, b)
            game = play_game(white, black, opening, game=2 * i + high_is_white)
            games.append(asdict(game))
            white_points = {"1-0": 1.0, "0-1": 0.0}.get(game.result, 0.5)
            points.append(white_points if high_is_white else 1 - white_points)
    for player in (a, b):
        if hasattr(player, "close"):
            player.close()
    n, score = len(points), sum(points) / len(points)
    half = 1.96 * math.sqrt(sum((p - score) ** 2 for p in points) / (n - 1) / n)
    return {"low": low, "high": high, "games": n, "score": score, "wins": points.count(1.0), "draws": points.count(0.5),
            "losses": points.count(0.0), "elo": elo(score), "elo_range": [elo(score - half), elo(score + half)],
            "ends": {e: sum(g["end"] == e for g in games) for e in sorted({g["end"] for g in games})},
            "median_plies": sorted(len(g["moves"]) for g in games)[n // 2], "records": games}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--players", default="random,grabber,sf1320,sf1", help=f"weakest first, from {', '.join(PLAYERS)}")
    parser.add_argument("--openings", type=int, default=100)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    names = args.players.split(",")
    openings = random_openings(args.openings, seed=1)
    jobs = [(low, high, openings) for low, high in itertools.combinations(names, 2)]
    with Pool(min(args.workers, len(jobs))) as pool:
        rows = pool.map(match, jobs)
    fmt = lambda x: "    —" if x is None else f"{x:+5.0f}"  # noqa: E731
    for r in rows:
        print(f"{r['high']:8s} vs {r['low']:8s}: {100 * r['score']:5.1f}%  (+{r['wins']} ={r['draws']} -{r['losses']})  "
              f"Elo {fmt(r['elo'])} [{fmt(r['elo_range'][0])}, {fmt(r['elo_range'][1])}]  "
              f"median {r['median_plies']} plies  {r['ends']}", flush=True)
    ratings = fit_ratings(names, rows)
    print("fitted together, " + names[0] + " = 0: " + ", ".join(f"{n} {ratings[n]:+.0f}" for n in names), flush=True)
    if args.out:  # the summary as JSON; the game records beside it, compressed (about 1 MB per 7,000 games)
        games_file = args.out.removesuffix(".json") + "-games.json.gz"
        records = {f"{r['high']} vs {r['low']}": [{**g, "moves": " ".join(g["moves"])} for g in r.pop("records")]
                   for r in rows}
        with gzip.open(games_file, "wt") as f:
            json.dump(records, f, separators=(",", ":"))
        with open(args.out, "w") as f:
            json.dump({"players": names, "ratings": ratings, "openings": openings, "pairings": rows}, f, indent=1)


if __name__ == "__main__":
    main()
