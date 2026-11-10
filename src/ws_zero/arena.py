"""The arena page (arena/): stand-in models, and the test cases its browser code must reproduce.

The browser plays the models with its own copies of the observation, the move numbering, the network and the
search (arena/js/). These cases, computed here with the python-chess reference game (which matches pgx exactly)
and ws_zero.net, are what those copies are checked against: in the page itself before a model is offered, and in
tests/test_arena_js.py.
"""
import random

import chess
import jax.numpy as jnp
import numpy as np

from ws_zero.game.moves import NUM_ACTIONS
from ws_zero.game.reference import SCALAR_PLANES, ReferenceGame
from ws_zero.net import apply, blank_params, init_params

ENCODING_LINES = [  # (starting FEN or None for the standard start, moves in UCI)
    (None, ["e2e4", "g8f6", "e4e5", "d7d5", "e5d6"]),  # en passant
    ("r3k2r/pppq1ppp/2npbn2/2b1p3/2B1P3/2NPBN2/PPPQ1PPP/R3K2R w KQkq - 0 8", ["e1g1", "e8c8"]),  # castling
    ("r1n1k3/1P6/8/8/8/8/6p1/4K2R w K - 0 1", ["b7c8n", "g2h1q"]),  # promotion and underpromotion
    (None, ["g1f3", "g8f6", "f3g1", "f6g8", "g1f3", "g8f6", "f3g1", "f6g8"]),  # threefold repetition
    ("6k1/5ppp/8/8/8/8/5PPP/R5K1 w - - 0 1", ["a1a8"]),  # checkmate
    ("7k/8/5QK1/8/8/8/8/8 w - - 0 1", ["f6f7"]),  # stalemate
]


def pawn_pusher():
    """Phase 3's hand-written network: three nonzero weights that make it prefer pushing a pawn two squares."""
    params = blank_params()  # every weight 0, every batch norm the identity
    params["stem/w"] = params["stem/w"].at[1, 1, 0, 0].set(1.0)
    params["policy/conv/w"] = params["policy/conv/w"].at[1, 1, 0, 0].set(1.0)
    params["policy/out/w"] = params["policy/out/w"].at[0, 0, 0, 17].set(1.0)
    return params


def material_counter(params, obs):
    """The Phase 4 gate's stand-in for a network: every move equally likely; value tanh((own material - the
    opponent's) / 4) for the side to move, pawn 1, knight and bishop 3, rook 5, queen 9. `params` is unused."""
    values = jnp.array([1.0, 3.0, 3.0, 5.0, 9.0, 0.0])
    mine, theirs = jnp.einsum("bhwp,p->b", obs[..., 0:6], values), jnp.einsum("bhwp,p->b", obs[..., 6:12], values)
    return jnp.zeros((obs.shape[0], NUM_ACTIONS)), jnp.tanh((mine - theirs) / 4)


def untrained():
    """The random arm's network before any training (seed 0): uniform move probabilities, value 0."""
    import jax
    return init_params(jax.random.key(0))


def random_lines(count: int, seed: int, max_plies: int = 512):
    """Random games from the start, played to their end under the reference rules."""
    rng = random.Random(seed)
    lines = []
    for _ in range(count):
        game, moves = ReferenceGame(repetition_key="pgx"), []
        while not game.outcome()[0] and len(moves) < max_plies:
            move = rng.choice(sorted(game.board.legal_moves, key=lambda m: m.uci()))
            game.push(move)
            moves.append(move.uci())
        lines.append((None, moves))
    return lines


def replay(fen, moves) -> ReferenceGame:
    game = ReferenceGame(fen, repetition_key="pgx")
    for uci in moves:
        game.push(chess.Move.from_uci(uci))
    return game


def encoding_case(fen, moves) -> dict:
    """What the browser must compute for the position after `moves`: legal actions, observation, outcome."""
    game = replay(fen, moves)
    obs = game.observation()
    binary = obs.copy()
    binary[..., list(SCALAR_PLANES)] = 0
    over, value = game.outcome()
    return {"fen": fen, "moves": moves, "legal": game.legal_actions(),
            "ones": np.flatnonzero(binary.reshape(-1)).tolist(),
            "scalars": [float(obs[0, 0, p]) for p in SCALAR_PLANES], "over": over, "value": value}


def encoding_cases(random_games: int = 3, every: int = 7) -> list[dict]:
    cases = []
    for fen, moves in ENCODING_LINES:
        cases += [encoding_case(fen, moves[:n]) for n in range(len(moves) + 1)]
    for fen, moves in random_lines(random_games, seed=0):
        cases += [encoding_case(fen, moves[:n]) for n in list(range(0, len(moves), every)) + [len(moves)]]
    return cases


def network_cases(params, lines) -> list[dict]:
    """The network's logits (for the legal moves, in action order) and value at the end of each line."""
    games = [replay(fen, moves) for fen, moves in lines]
    obs = jnp.asarray(np.stack([g.observation() for g in games]))
    logits, values = apply(params, obs)
    logits, values = np.asarray(logits), np.asarray(values)
    return [{"fen": fen, "moves": moves, "legal": g.legal_actions(),
             "logits": [float(logits[i, a]) for a in g.legal_actions()], "value": float(values[i])}
            for i, ((fen, moves), g) in enumerate(zip(lines, games))]


def network_lines() -> list[tuple]:
    """A few positions for the network checks: the start, Black to move, castling rights, a long game's end."""
    lines = [(None, []), (None, ["e2e4"]), (None, ["e2e4", "e7e5", "g1f3", "b8c6", "f1c4", "g8f6"])]
    lines += [(fen, moves) for fen, moves in ENCODING_LINES[:3]]
    fen, moves = random_lines(1, seed=1)[0]
    return lines + [(fen, moves[:40]), (fen, moves[:41])]

