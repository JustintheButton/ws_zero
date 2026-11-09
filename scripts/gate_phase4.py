"""Phase 4 gate: with a uniform dummy network that counts material, the search finds mate in one and does not hang its
queen, at 800 simulations, the project's setting (protocol §3 and §10, Phase 4 gate).

The dummy network: every legal move equally likely; value for the side to move = tanh(material difference / 4),
pawn 1, knight 3, bishop 3, rook 5, queen 9 (the scale of the Phase 3 checks). The search runs as in evaluation
games: no root noise, the most-visited move. Each position is searched with several seeds (mctx breaks exact ties
between moves with tiny random noise, and a uniform policy makes ties common); every run must pass.

Mate in one: pass if the move played gives checkmate. Hanging queen: the move fails if afterwards the side's queen can
be taken by a cheaper piece, or taken at all while undefended. python-chess checks that every position is what it
claims (a mate exists; a queen-hanging move and a safe move both exist).

    uv run python scripts/gate_phase4.py --out docs/gates/phase-4-gate.json
"""
import argparse
import json
import time

import chess
import jax
import jax.numpy as jnp
import numpy as np

from ws_zero.game.env import ChessEnv, state_from_fen
from ws_zero.game.moves import NUM_ACTIONS, action_to_move
from ws_zero.search import search_moves

PIECE_VALUES = jnp.array([1.0, 3.0, 3.0, 5.0, 9.0, 0.0])  # P N B R Q K
VALUE = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 100}

MATE_IN_ONE = {
    "back rank": "6k1/5ppp/8/8/8/8/5PPP/R5K1 w - - 0 1",
    "Scholar's mate": "r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4",
    "smothered mate": "6rk/6pp/8/6N1/8/8/8/6K1 w - - 0 1",
    "back rank, Black to move": "r5k1/5ppp/8/8/8/8/5PPP/6K1 b - - 0 1",
    "queen and king": "k7/8/1K6/8/8/8/8/6Q1 w - - 0 1",
    "mate by promotion": "7k/4P3/6K1/8/8/8/8/8 w - - 0 1",
}
HANGING_QUEEN = {
    "queen attacked by a pawn": "rnbqkbnr/pppp1ppp/8/4p3/3Q4/8/PPP1PPPP/RNB1KBNR w KQkq - 0 3",
    "poisoned f7 pawn": "rnbqkbnr/ppp2ppp/3p4/4p3/4P3/5Q2/PPPP1PPP/RNB1KBNR w KQkq - 0 3",
    "queen attacked by a knight": "r1bqkbnr/pppp1ppp/8/4n3/4P3/5Q2/PPPP1PPP/RNB1KBNR w KQkq - 0 4",
    "queen attacked by a pawn, Black to move": "rnb1kbnr/ppp1pppp/8/3q4/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 2",
}


def material_dummy(params, obs):
    """Uniform move logits; value tanh((my material - their material) / 4) for the side to move."""
    mine = jnp.einsum("bhwp,p->b", obs[..., 0:6], PIECE_VALUES)
    theirs = jnp.einsum("bhwp,p->b", obs[..., 6:12], PIECE_VALUES)
    return jnp.zeros((obs.shape[0], NUM_ACTIONS)), jnp.tanh((mine - theirs) / 4)


def hangs_queen(board: chess.Board, move: chess.Move) -> bool:
    """After `move`, can the mover's queen be taken by a cheaper piece, or taken at all while undefended?"""
    after = board.copy()
    after.push(move)
    us = board.turn
    for square in after.pieces(chess.QUEEN, us):
        attackers = after.attackers(not us, square)
        if not attackers:
            continue
        if not after.attackers(us, square):
            return True
        if min(VALUE[after.piece_type_at(a)] for a in attackers) < VALUE[chess.QUEEN]:
            return True
    return False


def check_positions():
    for name, fen in MATE_IN_ONE.items():
        board = chess.Board(fen)
        assert board.is_valid(), name
        assert not board.is_game_over(), name
        assert any(_gives_mate(board, m) for m in board.legal_moves), f"{name}: no mate in one"
    for name, fen in HANGING_QUEEN.items():
        board = chess.Board(fen)
        assert board.is_valid(), name
        hanging = [m for m in board.legal_moves if hangs_queen(board, m)]
        safe = [m for m in board.legal_moves if not hangs_queen(board, m)]
        assert hanging and safe, f"{name}: hanging {len(hanging)}, safe {len(safe)}"


def _gives_mate(board, move):
    after = board.copy()
    after.push(move)
    return after.is_checkmate()


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--simulations", type=int, default=800)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    check_positions()
    env = ChessEnv()
    positions = [("mate in one", name, fen) for name, fen in MATE_IN_ONE.items()] + \
                [("hanging queen", name, fen) for name, fen in HANGING_QUEEN.items()]
    state = jax.tree.map(lambda *xs: jnp.stack(xs), *[state_from_fen(fen) for _, _, fen in positions])
    search = jax.jit(lambda key: search_moves(env, material_dummy, None, key, state, args.simulations,
                                              self_play=False))
    results, start = [], time.time()
    for seed in range(args.seeds):
        actions, out = search(jax.random.key(seed))
        visits = np.asarray(out.search_tree.summary().visit_counts)
        qvalues = np.asarray(out.search_tree.summary().qvalues)
        for i, (kind, name, fen) in enumerate(positions):
            board = chess.Board(fen)
            move = action_to_move(board, int(actions[i]))
            passed = _gives_mate(board, move) if kind == "mate in one" else not hangs_queen(board, move)
            top = [a for a in np.argsort(-visits[i])[:3] if visits[i, a] > 0]
            results.append({
                "kind": kind, "position": name, "fen": fen, "seed": seed, "move": board.san(move), "passed": passed,
                "top_moves": [{"move": board.san(action_to_move(board, int(a))), "visits": int(visits[i, a]),
                               "value": round(float(qvalues[i, a]), 3)} for a in top]})
    summary = {"simulations": args.simulations, "seeds": args.seeds, "passed": all(r["passed"] for r in results),
               "seconds": round(time.time() - start, 1), "results": results}
    for kind, name, _ in positions:
        runs = [r for r in results if r["position"] == name]
        moves = sorted({r["move"] for r in runs})
        top = runs[0]["top_moves"]
        print(f"{'PASS' if all(r['passed'] for r in runs) else 'FAIL'}  {kind:13s}  {name:40s}  played {', '.join(moves):12s}"
              f"  top: " + ", ".join(f"{t['move']} {t['visits']} ({t['value']:+.2f})" for t in top))
    print(f"gate {'PASSED' if summary['passed'] else 'NOT PASSED'}: {sum(r['passed'] for r in results)}/{len(results)} runs")
    if args.out:
        with open(args.out, "w") as f:
            json.dump(summary, f, indent=1)


if __name__ == "__main__":
    main()
