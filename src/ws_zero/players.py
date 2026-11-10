"""The weak anchors of the evaluation ladder (protocol §8.2) and the loop that plays a game between two players.

A player has a `name`, `new_game(game)` (called before each game with the game's number) and `move(board)`, which
returns its move in a python-chess board that carries the game's moves. `ws_zero.stockfish.Stockfish` is one too.

The random players draw their moves from a generator seeded with (the player's seed, the game's number), so a game
replays exactly whatever was played before it.

Games follow the rules the networks train under (`ReferenceGame` with pgx's repetition rule): checkmate, stalemate,
100 plies without a capture or pawn move, insufficient material, a position occurring three times, and the 512-ply
cap, all automatic.
"""
import random
from dataclasses import dataclass

import chess

from ws_zero.game.reference import MAX_PLIES, ReferenceGame

PIECE_VALUE = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9}


def _legal(board: chess.Board) -> list[chess.Move]:
    return sorted(board.legal_moves, key=chess.Move.uci)  # a fixed order, whatever python-chess generates


class RandomMover:
    """Every legal move equally likely."""
    name = "Random mover"

    def __init__(self, seed: int = 0):
        self.seed = seed
        self.new_game(0)

    def new_game(self, game: int = 0) -> None:
        self.rng = random.Random(f"{self.seed}:{game}")

    def move(self, board: chess.Board) -> chess.Move:
        return self.rng.choice(_legal(board))


class MaterialGrabber(RandomMover):
    """Captures the most valuable piece available (pawn 1, knight and bishop 3, rook 5, queen 9; en passant takes a
    pawn), choosing at random between equally valuable captures; with nothing to take, a random move."""
    name = "Material-grabber"

    def move(self, board: chess.Board) -> chess.Move:
        moves = _legal(board)
        values = [self._victim(board, m) for m in moves]
        best = max(values)
        return self.rng.choice([m for m, v in zip(moves, values) if v == best])

    @staticmethod
    def _victim(board: chess.Board, move: chess.Move) -> int:
        if board.is_en_passant(move):
            return PIECE_VALUE[chess.PAWN]
        piece = board.piece_at(move.to_square)
        return PIECE_VALUE[piece.piece_type] if piece else 0


class Diluted:
    """A base player that, on each move, plays a random legal move with probability `random_share` and otherwise its
    own move. The dice are seeded with (seed, game number), like the random players'."""

    def __init__(self, base, random_share: float, seed: int = 0):
        self.base, self.random_share, self.seed = base, random_share, seed
        self.name = f"{base.name}, {round(100 * random_share)}% random moves"
        self.rng = random.Random(f"{seed}:0")

    def new_game(self, game: int = 0) -> None:
        self.rng = random.Random(f"{self.seed}:{game}")
        self.base.new_game(game)

    def move(self, board: chess.Board) -> chess.Move:
        if self.rng.random() < self.random_share:
            return self.rng.choice(_legal(board))
        return self.base.move(board)

    def close(self) -> None:
        if hasattr(self.base, "close"):
            self.base.close()


@dataclass
class Game:
    white: str
    black: str
    start_fen: str | None  # None: the start position
    moves: list[str]  # UCI, the opening included
    result: str  # "1-0", "0-1" or "1/2-1/2"
    end: str  # checkmate, stalemate, fifty-move rule, insufficient material, threefold repetition, ply cap


def play_game(white, black, opening=(), *, game: int = 0, start_fen: str | None = None,
              max_plies: int = MAX_PLIES) -> Game:
    """Play one game from `start_fen` (default: the start position) after the `opening` moves (UCI). The opening
    counts towards the ply cap. `game` is the game's number, which seeds the random players."""
    ref = ReferenceGame(start_fen, max_plies=max_plies, repetition_key="pgx")
    for uci in opening:
        move = chess.Move.from_uci(uci)
        if move not in ref.board.legal_moves:
            raise ValueError(f"illegal opening move {uci} in {ref.board.fen()}")
        ref.push(move)
    white.new_game(game)
    black.new_game(game)
    while (end := ref.end_reason()) is None:
        player = white if ref.board.turn == chess.WHITE else black
        move = player.move(ref.board.copy())
        if move not in ref.board.legal_moves:
            raise ValueError(f"{player.name} played an illegal move, {move}, in {ref.board.fen()}")
        ref.push(move)
    result = "1/2-1/2" if end != "checkmate" else ("0-1" if ref.board.turn == chess.WHITE else "1-0")
    return Game(white.name, black.name, start_fen, [m.uci() for m in ref.board.move_stack], result, end)
