"""Phase 5 gate: two identical checkpoints rate as equal within error bars (protocol §10: tested with the opening
suite, since otherwise identical deterministic agents replay identical games).

Each agent was entered twice, under two names with two search seeds, and each copy played every opponent of the
ladder on the checkpoint openings (scripts/modal_evaluate.py). The ladder's opponents played each other on the same
openings (the calibration, scripts/ladder_matches.py). All games are fitted together (ws_zero.ratings, random mover
at 0); the 95% range of each rating and of the gap between the two copies comes from 500 refits on redrawn games.
The gate passes if, for every agent, the range of the gap between its two copies includes 0.

    uv run python scripts/gate_phase5.py --out docs/gates/phase-5-gate.json
"""
import argparse
import json
from pathlib import Path

import numpy as np

from ws_zero.ratings import fit_ratings, resampled_ratings

GATES = Path(__file__).resolve().parents[1] / "docs" / "gates"


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--calibration", type=Path, default=GATES / "phase-5-calibration.json")
    parser.add_argument("--agents", default="untrained,material")
    parser.add_argument("--seeds", default="0,1")
    parser.add_argument("--samples", type=int, default=500)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    calibration = json.loads(args.calibration.read_text())
    names = list(calibration["players"])
    pairings = [{k: p[k] for k in ("low", "high", "games", "score", "wins", "draws", "losses")}
                for p in calibration["pairings"]]
    copies, evaluations = {}, {}
    for agent in args.agents.split(","):
        copies[agent] = []
        for seed in args.seeds.split(","):
            e = json.loads((GATES / f"phase-5-eval-{agent}-seed{seed}.json").read_text())
            copies[agent].append(e["name"])
            evaluations[e["name"]] = e
            names.append(e["name"])
            pairings += [{"low": o["opponent"], "high": e["name"], "games": o["games"], "score": o["score"],
                          "wins": o["wins"], "draws": o["draws"], "losses": o["losses"]} for o in e["opponents"]]
    ratings = fit_ratings(names, pairings)
    samples = resampled_ratings(names, pairings, samples=args.samples, seed=0)
    span = lambda x: [float(v) for v in np.percentile(x, [2.5, 97.5])]  # noqa: E731

    print(f"ratings, {names[0]} = 0 (95% range):")
    for name in names:
        lo, hi = span(samples[name])
        print(f"  {name:18s} {ratings[name]:7.0f}   [{lo:6.0f}, {hi:6.0f}]")
    results, passed = [], True
    for agent, (a, b) in copies.items():
        lo, hi = span(samples[a] - samples[b])
        ok = lo <= 0 <= hi
        passed &= ok
        table = [{"opponent": x["opponent"], a: x["score"], b: y["score"]}
                 for x, y in zip(evaluations[a]["opponents"], evaluations[b]["opponents"])]
        results.append({"agent": agent, "copies": [a, b], "ratings": [ratings[a], ratings[b]],
                        "gap": ratings[a] - ratings[b], "gap_range": [lo, hi], "passed": ok, "scores": table})
        print(f"\n{agent}: {a} {ratings[a]:.0f}, {b} {ratings[b]:.0f}; gap {ratings[a] - ratings[b]:+.0f} "
              f"[{lo:+.0f}, {hi:+.0f}]  {'PASS' if ok else 'FAIL'}")
        print("  score against each opponent (copy 1, copy 2): " +
              ", ".join(f"{t['opponent']} {100 * t[a]:.0f}%/{100 * t[b]:.0f}%" for t in table))
    print(f"\ngate {'PASSED' if passed else 'NOT PASSED'}")
    if args.out:
        with open(args.out, "w") as f:
            json.dump({"passed": passed, "samples": args.samples, "players": names, "ratings": ratings,
                       "ranges": {n: span(samples[n]) for n in names}, "agents": results,
                       "evaluation_seconds": {n: e["seconds"] for n, e in evaluations.items()}}, f, indent=1)
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
