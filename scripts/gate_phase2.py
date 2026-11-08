"""Phase 2 gate: pgx (via ChessEnv) against the python-chess reference, over many random games.

At every position of every game it checks that:
- the legal actions are identical;
- every legal move round-trips (move -> action -> move) and every legal action round-trips (action -> move -> action);
- the 119-plane observations are identical (scalar planes 113 and 118 within 1e-6, for XLA rounding);
- the terminal flag and value are identical.
A game stops at the first legal-move or outcome mismatch (so one bug does not cascade). Moves are uniformly random.

--repetition-key fide (default) compares with FIDE repetition rules; --repetition-key pgx makes the reference use
pgx's convention (an en passant square counts after every double push), which isolates all other differences.

Usage: uv run python scripts/gate_phase2.py --games 2000 --batch 250 --seed 0 [--repetition-key pgx] [--out f.json]
"""
import argparse
import json
import random
import time
from collections import Counter

import chess
import numpy as np

from ws_zero.game.env import ChessEnv
from ws_zero.game.moves import action_to_move, move_to_action
from ws_zero.game.reference import ReferenceGame, observations_match


def move_kind(board: chess.Board, move: chess.Move) -> list[str]:
    kinds = []
    color = "white" if board.turn == chess.WHITE else "black"
    if board.is_kingside_castling(move):
        kinds.append(f"castle_kingside_{color}")
    elif board.is_queenside_castling(move):
        kinds.append(f"castle_queenside_{color}")
    if board.is_en_passant(move):
        kinds.append(f"en_passant_{color}")
    if move.promotion:
        kinds.append(f"promote_{chess.piece_name(move.promotion)}_{color}")
    return kinds


def ending(ref: ReferenceGame) -> str:
    b = ref.board
    if b.is_checkmate():
        return "checkmate"
    if b.is_stalemate():
        return "stalemate"
    if b.is_insufficient_material():
        return "insufficient_material"
    if b.halfmove_clock >= 100:
        return "fifty_move_rule"
    if b.is_repetition(3):
        return "threefold_repetition"
    return "ply_cap"


def run(games: int, batch: int, seed: int, repetition_key: str = "fide") -> dict:
    env, rng = ChessEnv(), random.Random(seed)
    mismatches, examples = Counter(), []
    legal_seen, played, endings = Counter(), Counter(), Counter()
    positions = 0
    t0 = time.time()

    def record(kind, ref, detail):
        mismatches[kind] += 1
        if len(examples) < 20:
            examples.append({"kind": kind, "fen": ref.board.fen(), "ply": ref.plies, "detail": detail})

    for start in range(0, games, batch):
        n = min(batch, games - start)
        refs = [ReferenceGame(repetition_key=repetition_key) for _ in range(n)]
        state, done = env.init(n), np.zeros(n, dtype=bool)
        while not done.all():
            masks = np.asarray(env.legal_action_mask(state))
            obs = np.asarray(env.observe(state))
            term = np.asarray(env.is_terminal(state))
            value = np.asarray(env.terminal_value(state))
            actions = np.zeros(n, dtype=np.int32)
            for i in np.flatnonzero(~done):
                ref, board = refs[i], refs[i].board
                positions += 1
                legal_moves = list(board.legal_moves)
                ours, theirs = ref.legal_actions(), np.flatnonzero(masks[i]).tolist()
                if ours != theirs:
                    record("legal_actions", ref, {"only_reference": sorted(set(ours) - set(theirs)),
                                                  "only_pgx": sorted(set(theirs) - set(ours))})
                    done[i] = True
                    continue
                for move in legal_moves:
                    for kind in move_kind(board, move):
                        legal_seen[kind] += 1
                    if action_to_move(board, move_to_action(board, move)) != move:
                        record("move_round_trip", ref, move.uci())
                for action in theirs:
                    if move_to_action(board, action_to_move(board, action)) != action:
                        record("action_round_trip", ref, int(action))
                expected = ref.observation()
                if not observations_match(obs[i], expected):
                    planes = sorted(set(np.argwhere(~np.isclose(obs[i], expected, atol=1e-6, rtol=0))[:, 2].tolist()))
                    record("observation", ref, {"planes": planes})
                over, ref_value = ref.outcome()
                if (over, ref_value) != (bool(term[i]), float(value[i])):
                    record("outcome", ref, {"reference": [over, ref_value], "pgx": [bool(term[i]), float(value[i])]})
                    done[i] = True
                    continue
                if over:
                    endings[ending(ref)] += 1
                    done[i] = True
                    continue
                move = rng.choice(legal_moves)
                for kind in move_kind(board, move):
                    played[kind] += 1
                actions[i] = move_to_action(board, move)
                ref.push(move)
            state = env.step(state, actions)
        print(f"  {start + n}/{games} games, {positions:,} positions, {sum(mismatches.values())} mismatches, "
              f"{time.time() - t0:.0f} s", flush=True)

    return {
        "games": games, "seed": seed, "repetition_key": repetition_key, "positions_checked": positions,
        "mismatches": dict(mismatches), "mismatch_examples": examples,
        "endings": dict(endings), "legal_move_occurrences": dict(sorted(legal_seen.items())),
        "moves_played": dict(sorted(played.items())), "seconds": round(time.time() - t0, 1),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--games", type=int, default=2000)
    parser.add_argument("--batch", type=int, default=250)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--repetition-key", choices=["fide", "pgx"], default="fide")
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()
    summary = run(args.games, args.batch, args.seed, args.repetition_key)
    text = json.dumps(summary, indent=2)
    print(text)
    if args.out:
        with open(args.out, "w") as f:
            f.write(text + "\n")
