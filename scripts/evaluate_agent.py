"""Phase 5: evaluate one agent (a network with search) against the ladder (protocol §8.2-8.3).

Against each opponent it plays every opening of the suite twice, once with each colour: 100 games per opponent with
the checkpoint suite. Reports the agent's score against each opponent and how long each took. Needs a GPU for 800
simulations; run it on Modal:

    uv run modal run scripts/modal_evaluate.py --jobs untrained:0,untrained:1

Agents so far: `untrained` (the random arm's network before training) and `material` (the Phase 4 stand-in: uniform
move probabilities, material-count value). Trained checkpoints are added in Phase 7.
"""
import argparse
import gzip
import json
import math
import time
from dataclasses import asdict

import jax

from ws_zero import openings as suites
from ws_zero.arena import material_counter, untrained
from ws_zero.evaluate import make_agent, play_agent_games
from ws_zero.game.env import ChessEnv
from ws_zero.ladder import LADDER, opponent
from ws_zero.net import apply

AGENTS = {"untrained": lambda: (apply, untrained()), "material": lambda: (material_counter, None)}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--agent", default="untrained", choices=sorted(AGENTS))
    parser.add_argument("--name", default=None, help="the agent's name in the records (default: agent and seed)")
    parser.add_argument("--simulations", type=int, default=800)
    parser.add_argument("--opponents", default=",".join(LADDER))
    parser.add_argument("--suite", default="checkpoint", choices=["checkpoint", "go_no_go"])
    parser.add_argument("--openings", type=int, default=None, help="only the first so many openings (for a quick try)")
    parser.add_argument("--seed", type=int, default=0, help="seeds the search's tie-breaking")
    parser.add_argument("--threads", type=int, default=8, help="Stockfish engines per opponent, and threads")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    name = args.name or f"{args.agent}-seed{args.seed}"
    openings = [o["moves"] for o in suites.load()[args.suite]][:args.openings]
    env = ChessEnv()
    evaluate, params = AGENTS[args.agent]()
    agent = make_agent(env, evaluate, params, args.simulations)
    rows, records, start = [], {}, time.time()
    for opponent_name in args.opponents.split(","):
        make, close = opponent(opponent_name, engines=args.threads)
        began = time.time()
        games = play_agent_games(env, evaluate, params, args.simulations, make, openings, agent_name=name,
                                 seed=args.seed, threads=args.threads, agent=agent)
        close()
        points = [{"1/2-1/2": 0.5}.get(g.result, float((g.result == "1-0") == (g.white == name))) for g in games]
        n, score = len(points), sum(points) / len(points)
        half = 1.96 * math.sqrt(sum((p - score) ** 2 for p in points) / (n - 1) / n) if n > 1 else 0.0
        rows.append({"opponent": opponent_name, "games": n, "score": score, "range": [score - half, score + half],
                     "wins": points.count(1.0), "draws": points.count(0.5), "losses": points.count(0.0),
                     "seconds": time.time() - began, "longest_game_plies": max(len(g.moves) for g in games),
                     "ends": {e: sum(g.end == e for g in games) for e in sorted({g.end for g in games})}})
        records[opponent_name] = [{**asdict(g), "moves": " ".join(g.moves)} for g in games]
        r = rows[-1]
        print(f"{name} vs {opponent_name:8s}: {100 * score:5.1f}%  (+{r['wins']} ={r['draws']} -{r['losses']})  "
              f"{r['seconds']:5.0f} s  longest game {r['longest_game_plies']} plies  {r['ends']}", flush=True)
    total = time.time() - start
    print(f"{name}: {sum(r['games'] for r in rows)} games in {total:.0f} s on {jax.devices()[0].device_kind}", flush=True)
    if args.out:
        with gzip.open(args.out.removesuffix(".json") + "-games.json.gz", "wt") as f:
            json.dump(records, f, separators=(",", ":"))
        with open(args.out, "w") as f:
            json.dump({"agent": args.agent, "name": name, "simulations": args.simulations, "suite": args.suite,
                       "seed": args.seed, "device": jax.devices()[0].device_kind, "seconds": total,
                       "opponents": rows}, f, indent=1)


if __name__ == "__main__":
    main()
