"""Build the lab notebook page (lab/template.html) with positions, observations and legal moves from pgx.

Every position is checked against the python-chess reference before it goes on the page.

    uv run python scripts/build_lab.py --out lab/build/lab.html
"""
import argparse
import json
import subprocess
from pathlib import Path

import chess
import jax
import jax.numpy as jnp
import numpy as np

from ws_zero.game.env import ChessEnv, initial_state, state_from_fen
from ws_zero.game.moves import move_to_action
from ws_zero.game.reference import SCALAR_PLANES, ReferenceGame, observations_match
from ws_zero.net import apply, blank_params, masked_log_softmax

ROOT = Path(__file__).resolve().parents[1]

OPERA_GAME = ("e4 e5 Nf3 d6 d4 Bg4 dxe5 Bxf3 Qxf3 dxe5 Bc4 Nf6 Qb3 Qe7 Nc3 c6 Bg5 b5 Nxb5 cxb5 Bxb5+ Nbd7 "
              "O-O-O Rd8 Rxd7 Rxd7 Rd1 Qe6 Bxd7+ Nxd7 Qb8+ Nxb8 Rd8#").split()

SCENES = [  # (name, starting FEN or None for the start position, moves in SAN)
    ("opera", None, OPERA_GAME),
    ("castling", "r3k2r/pppq1ppp/2npbn2/2b1p3/2B1P3/2NPBN2/PPPQ1PPP/R3K2R w KQkq - 0 8", ["O-O", "O-O-O"]),
    ("enpassant", None, ["e4", "Nf6", "e5", "d5", "exd6"]),
    ("promotion", "r1n1k3/1P6/8/8/8/8/6p1/4K2R w K - 0 1", ["bxc8=N", "gxh1=Q+"]),
]


def hand_written_network():
    """Three nonzero weights: copy "my pawns" through the stem and policy conv, then score "two squares forward"."""
    params = blank_params()  # every weight 0, every batch norm the identity
    params["stem/w"] = params["stem/w"].at[1, 1, 0, 0].set(1.0)
    params["policy/conv/w"] = params["policy/conv/w"].at[1, 1, 0, 0].set(1.0)
    params["policy/out/w"] = params["policy/out/w"].at[0, 0, 0, 17].set(1.0)
    return params


HAND = hand_written_network()


def plane_description(plane: int) -> str:
    """Plain-language meaning of one of the 73 move types (docs/encoding.md)."""
    if plane < 9:
        piece = ["rook", "bishop", "knight"][plane // 3]
        direction = ["straight", "capture toward h", "capture toward a"][plane % 3]
        return f"underpromotion to {piece}, {direction}"
    i = plane - 9
    if i >= 56:
        k = i - 56
        d_rank = [-1, +1, -2, +2, -1, +1, -2, +2][k]
        d_file = [-2, -2, -1, -1, +2, +2, +1, +1][k]
        return f"knight jump ({d_rank:+d} rank, {d_file:+d} file)"
    group, n = divmod(i, 7)
    names = ["backward", "forward", "toward the a-file", "toward the h-file",
             "backward toward a", "forward toward h", "forward toward a", "backward toward h"]
    distance = 7 - n if group in (0, 2, 4, 6) else n + 1
    return f"queen-style: {distance} square{'s' if distance > 1 else ''} {names[group]}"


def encode_observation(obs: np.ndarray) -> list:
    """Binary planes as 16 hex digits (squares in row-major order, first square = high bit); scalars as numbers."""
    planes = []
    for p in range(obs.shape[-1]):
        if p in SCALAR_PLANES:
            planes.append(round(float(obs[0, 0, p]), 6))
        else:
            bits = obs[:, :, p].reshape(-1).astype(int)
            planes.append("".join(f"{int(''.join(map(str, bits[k:k + 4])), 2):x}" for k in range(0, 64, 4)))
    return planes


def build_scene(env: ChessEnv, name: str, fen: str | None, sans: list[str]) -> dict:
    ref = ReferenceGame(fen, repetition_key="pgx")
    state = jax.tree.map(lambda x: x[None], initial_state() if fen is None else state_from_fen(fen))
    positions, last = [], None
    for ply in range(len(sans) + 1):
        board = ref.board
        obs = np.asarray(env.observe(state))[0]
        mask = np.asarray(env.legal_action_mask(state))[0]
        assert observations_match(obs, ref.observation()), f"{name}, ply {ply}: observation differs"
        assert np.flatnonzero(mask).tolist() == ref.legal_actions(), f"{name}, ply {ply}: legal moves differ"
        moves = sorted(([m.uci(), board.san(m), move_to_action(board, m)] for m in board.legal_moves),
                       key=lambda m: m[2])
        terminal, value = ref.outcome()
        logp = np.asarray(masked_log_softmax(apply(HAND, jnp.asarray(obs)[None])[0], jnp.asarray(mask)[None]))[0]
        hand = [round(float(np.exp(logp[m[2]])), 4) for m in moves]
        positions.append({"fen": board.fen(), "ply": ref.plies, "last": last, "obs": encode_observation(obs),
                          "moves": moves, "hand": hand, "check": board.is_check(), "terminal": terminal,
                          "value": value})
        if ply < len(sans):
            move = board.parse_san(sans[ply])
            last = {"san": sans[ply], "uci": move.uci(), "action": move_to_action(board, move)}
            state = env.step(state, [last["action"]])
            ref.push(move)
    return {"name": name, "fen": fen, "positions": positions}


def phase3_data(gates: Path) -> dict:
    """Value-head check curves (no batch norm: original and linear head; batch norm), Modal cost per 1,000 games,
    the step-cap table, and the gate verdict (current design: batch norm)."""
    def curves(name):
        path = gates / name
        if not path.exists():
            return []
        return [[{"step": c["step"], "mse": round(c["value_mse"], 4)} for c in r["curve"]]
                for r in json.loads(path.read_text())["results"]]

    gpus = []
    bench = gates / "phase-3-bench-modal.json"
    if bench.exists():
        for gpu, r in json.loads(bench.read_text())["results"].items():
            if "error" in r:
                continue
            one = {e["chunk_plies"]: e for e in r["estimates"] if e["batch"] == 100}
            gpus.append({"gpu": gpu, "perHour": round(r["usd_per_hour"], 2),
                         "cost": [round(one[340]["usd_per_1000_games"], 2), round(one[512]["usd_per_1000_games"], 2)],
                         "gamesPerHour": [round(one[512]["games_per_hour"]), round(one[340]["games_per_hour"])]})
    gate = [json.loads((gates / f"phase-3-overfit-realistic-seed{s}.json").read_text())["passed"]
            for s in range(3) if (gates / f"phase-3-overfit-realistic-seed{s}.json").exists()]
    return {"valueOriginal": curves("phase-3-value-check-original-head.json"),
            "valueFixed": curves("phase-3-value-check-no-bn.json"), "valueBN": curves("phase-3-value-check.json"),
            "gpus": sorted(gpus, key=lambda g: g["cost"][1]),
            "gatePassed": bool(gate) and all(gate), "stability": stability_rows(gates), "lrs": lr_rows(gates)}


def lr_rows(gates: Path) -> list:
    """AlphaZero's value head at each D1 learning rate (realistic check, 1,000 steps), without and with warmup:
    for each, the runs and how many shut off at some point and how many were still broken at the end."""
    path = gates / "phase-3-lr-diagnostics.json"
    if not path.exists():
        return []
    runs = [r for r in json.loads(path.read_text())["results"] if r["value_head"].startswith("alphazero")]
    rows = []
    for lr in sorted({r["lr"] for r in runs}):
        row = {"lr": lr}
        for key, warm in (("cold", False), ("warm", True)):
            group = [r for r in runs if r["lr"] == lr and (r["warmup_steps"] > 0) == warm]
            row[key] = None if not group else {
                "runs": len(group), "broke": sum(r["first_broken_step"] is not None for r in group),
                "stuck": sum(r["broken_at_end"] for r in group),
                "mse": [r["value_mse_at_1000"] for r in group]}
        rows.append(row)
    return rows


def stability_rows(gates: Path) -> list:
    """One row per (design, test, step cap): runs stuck for good or briefly, final value error, move KL, gate verdict.

    design "nobn": protocol v1.0 without batch norm (linear value conv); "bn": batch norm restored, no cap.
    """
    def from_curve(design, mode, curve):
        return (design, mode, 0.0, any(c["saturated"] >= 0.5 for c in curve), curve[-1]["saturated"],
                curve[-1]["value_mse"], curve[-1]["policy_kl"])

    runs = []  # (design, mode, cap, ever_saturated, final_saturated, value_mse, kl)
    long_bn = gates / "phase-3-stability-bn.json"  # full-length gate runs with batch norm, if made
    for design, suffix in (("nobn", "-no-bn"), ("bn", "")):
        if design == "bn" and long_bn.exists():
            continue
        for s in range(3):
            path = gates / f"phase-3-overfit-realistic{suffix}-seed{s}.json"
            if path.exists():
                runs.append(from_curve(design, "gate", json.loads(path.read_text())["curve"]))
    path = gates / "phase-3-value-check.json"
    if path.exists():
        runs += [from_curve("bn", "check", r["curve"]) for r in json.loads(path.read_text())["results"]]
    for design, path in (("nobn", gates / "phase-3-stability.json"), ("bn", long_bn)):
        if path.exists():
            for r in json.loads(path.read_text())["results"]:
                runs.append((design, r["mode"], r["clip"], r["first_saturated_step"] is not None,
                             r["final_saturated"], r["final_value_mse"], r["final_move_kl"]))
    rows = []
    order = [("nobn", "check", 0.0), ("nobn", "check", 5.0), ("bn", "check", 0.0),
             ("nobn", "gate", 0.0), ("nobn", "gate", 5.0), ("nobn", "gate", 2.0), ("bn", "gate", 0.0)]
    for design, mode, cap in order:
        group = [r[3:] for r in runs if r[:3] == (design, mode, cap)]
        if not group:
            continue
        broke = sum(r[1] >= 0.5 for r in group)
        rows.append({"design": design, "mode": mode, "cap": cap, "runs": len(group), "broke": broke,
                     "wobbled": sum(r[0] for r in group) - broke, "mse": [round(r[2], 4) for r in group],
                     "kl": [round(min(r[3] for r in group), 3), round(max(r[3] for r in group), 3)],
                     "gate": None if mode == "check" else sum(r[3] < 0.05 and r[2] <= 0.01 and r[1] < 0.5
                                                              for r in group)})
    return rows


# The 200-simulation gate run of 8 Nov 2026 (untried moves at the position's value). Its results file was replaced
# by the 800-simulation run, so its pass counts and time are copied here from its printed summary.
GATE_200 = {"simulations": 200, "seconds": 138.6, "failed": {"queen attacked by a knight": 5}}


# One search of the 10 gate positions on the Mac (M4 Pro, CPU), measured 9 Nov 2026 while looking for the cause of the
# slow 800 run; "alone" is mctx with a do-nothing game of the same size (4,672 moves, no chess, no network). The 800 row
# is the gate run's time divided by its 5 runs.
SEARCH_TIMING = [{"sims": 100, "seconds": 4.3, "alone": 0.16}, {"sims": 200, "seconds": 26.8, "alone": 0.45},
                 {"sims": 400, "seconds": 186.9, "alone": 1.40}, {"sims": 800, "seconds": 1387.1, "estimated": True}]


def phase4_data(gates: Path) -> dict:
    """The Phase 4 gate positions: moves played in each run, pass counts at 200 and 800 simulations, the most-visited
    moves of the first run, and for the queen positions the attacked queen and its attackers."""
    path = gates / "phase-4-gate.json"
    if not path.exists():
        return {}
    gate = json.loads(path.read_text())
    positions = {}
    for r in gate["results"]:
        board = chess.Board(r["fen"])
        p = positions.setdefault(r["position"], {"name": r["position"], "kind": r["kind"], "fen": r["fen"], "runs": []})
        p["runs"].append({"move": r["move"], "uci": board.parse_san(r["move"]).uci(), "passed": r["passed"]})
        if r["seed"] == 0:
            p["top"] = [{"move": t["move"], "visits": t["visits"]} for t in r["top_moves"]]
    for p in positions.values():
        board = chess.Board(p["fen"])
        threat = []
        for square in board.pieces(chess.QUEEN, board.turn) if p["kind"] == "hanging queen" else []:
            attackers = list(board.attackers(not board.turn, square))
            if attackers:
                threat += [chess.square_name(s) for s in [square, *attackers]]
        p["threat"] = threat
        p["passed"] = {"200": len(p["runs"]) - GATE_200["failed"].get(p["name"], 0),
                       str(gate["simulations"]): sum(x["passed"] for x in p["runs"])}
    return {"simulations": gate["simulations"], "seeds": gate["seeds"], "positions": list(positions.values()),
            "scan": SEARCH_TIMING,
            "runs": [{"simulations": 200, "seconds": GATE_200["seconds"]},
                     {"simulations": gate["simulations"], "seconds": gate["seconds"]}]}


def commit_label() -> str:
    def git(*args):
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
    label = git("rev-parse", "--short", "HEAD")
    return label + (" (with uncommitted changes)" if git("status", "--porcelain") else "")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=ROOT / "lab" / "build" / "lab.html")
    args = parser.parse_args()

    env = ChessEnv()
    data = {"planes": [plane_description(p) for p in range(73)],
            "scenes": [build_scene(env, *scene) for scene in SCENES],
            "phase3": phase3_data(ROOT / "docs" / "gates"), "phase4": phase4_data(ROOT / "docs" / "gates")}
    page = (ROOT / "lab" / "template.html").read_text()
    page = page.replace("__COMMIT__", commit_label()).replace("__DATA__", json.dumps(data, separators=(",", ":")))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(page)
    print(f"wrote {args.out} ({len(page):,} bytes):",
          ", ".join(f"{s['name']} {len(s['positions'])} positions" for s in data["scenes"]))


if __name__ == "__main__":
    main()
