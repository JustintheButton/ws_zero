"""Build the lab notebook page (lab/template.html) with positions, observations and legal moves from pgx.

Every position is checked against the python-chess reference before it goes on the page.

    uv run python scripts/build_lab.py --out lab/build/lab.html
"""
import argparse
import json
import subprocess
from pathlib import Path

import jax
import numpy as np

from ws_zero.game.env import ChessEnv, initial_state, state_from_fen
from ws_zero.game.moves import move_to_action
from ws_zero.game.reference import SCALAR_PLANES, ReferenceGame, observations_match

ROOT = Path(__file__).resolve().parents[1]

OPERA_GAME = ("e4 e5 Nf3 d6 d4 Bg4 dxe5 Bxf3 Qxf3 dxe5 Bc4 Nf6 Qb3 Qe7 Nc3 c6 Bg5 b5 Nxb5 cxb5 Bxb5+ Nbd7 "
              "O-O-O Rd8 Rxd7 Rxd7 Rd1 Qe6 Bxd7+ Nxd7 Qb8+ Nxb8 Rd8#").split()

SCENES = [  # (name, starting FEN or None for the start position, moves in SAN)
    ("opera", None, OPERA_GAME),
    ("castling", "r3k2r/pppq1ppp/2npbn2/2b1p3/2B1P3/2NPBN2/PPPQ1PPP/R3K2R w KQkq - 0 8", ["O-O", "O-O-O"]),
    ("enpassant", None, ["e4", "Nf6", "e5", "d5", "exd6"]),
    ("promotion", "r1n1k3/1P6/8/8/8/8/6p1/4K2R w K - 0 1", ["bxc8=N", "gxh1=Q+"]),
]


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
        positions.append({"fen": board.fen(), "ply": ref.plies, "last": last, "obs": encode_observation(obs),
                          "moves": moves, "check": board.is_check(), "terminal": terminal, "value": value})
        if ply < len(sans):
            move = board.parse_san(sans[ply])
            last = {"san": sans[ply], "uci": move.uci(), "action": move_to_action(board, move)}
            state = env.step(state, [last["action"]])
            ref.push(move)
    return {"name": name, "fen": fen, "positions": positions}


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
            "scenes": [build_scene(env, *scene) for scene in SCENES]}
    page = (ROOT / "lab" / "template.html").read_text()
    page = page.replace("__COMMIT__", commit_label()).replace("__DATA__", json.dumps(data, separators=(",", ":")))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(page)
    print(f"wrote {args.out} ({len(page):,} bytes):",
          ", ".join(f"{s['name']} {len(s['positions'])} positions" for s in data["scenes"]))


if __name__ == "__main__":
    main()
