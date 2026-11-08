"""Reference chess game on python-chess, used to check pgx: legal actions, the 119-plane observation, and outcomes.

The observation follows AlphaZero's input features with pgx's exact conventions (see docs/encoding.md). The rules
follow python-chess (FIDE), with AlphaZero's automatic draws: threefold repetition, 100 plies without capture or
pawn move, insufficient material, and the ply cap.
"""
from collections import Counter

import chess
import numpy as np

from ws_zero.game.moves import move_to_action

HISTORY = 8
PLANES_PER_STEP = 14
NUM_PLANES = HISTORY * PLANES_PER_STEP + 7  # 119
MAX_PLIES = 512


SCALAR_PLANES = (113, 118)  # plies / 512 and no-progress / 100: continuous, may differ from pgx by rounding


def observations_match(a: np.ndarray, b: np.ndarray, atol: float = 1e-6) -> bool:
    """Exact on the 117 binary planes; within `atol` on the two scalar planes."""
    binary = np.ones(a.shape[-1], dtype=bool)
    binary[list(SCALAR_PLANES)] = False
    return bool(np.array_equal(a[..., binary], b[..., binary])
                and np.allclose(a[..., ~binary], b[..., ~binary], atol=atol, rtol=0))


def _position_key(board: chess.Board, repetition_key: str) -> str:
    """Repetition identity: pieces, side to move, castling rights and an en passant square.

    "fide": the en passant square counts only if a legal en passant capture exists (FIDE, python-chess).
    "pgx": it counts after every double pawn push, as in pgx's hash.
    """
    en_passant = {"fide": "legal", "pgx": "fen"}[repetition_key]
    return " ".join(board.fen(en_passant=en_passant).split()[:4])


class ReferenceGame:
    def __init__(self, fen: str | None = None, max_plies: int = MAX_PLIES, repetition_key: str = "fide"):
        self.board = chess.Board(fen) if fen else chess.Board()
        self.max_plies = max_plies
        self.repetition_key = repetition_key
        self.plies = 0
        self._snapshots = [self._snapshot()]  # one per position, oldest first
        self._key_counts = Counter(s[2] for s in self._snapshots)

    def _snapshot(self):
        pieces = [(sq, p.color, p.piece_type) for sq, p in self.board.piece_map().items()]
        return pieces, self.board.turn, _position_key(self.board, self.repetition_key)

    def push(self, move: chess.Move) -> None:
        self.board.push(move)
        self.plies += 1
        snap = self._snapshot()
        self._snapshots.append(snap)
        self._key_counts[snap[2]] += 1

    def legal_actions(self) -> list[int]:
        return sorted(move_to_action(self.board, m) for m in self.board.legal_moves)

    def observation(self) -> np.ndarray:
        """(8, 8, 119) float32, from the side to move's perspective; row 0 is the far rank, column 0 the a-file."""
        turn = self.board.turn
        obs = np.zeros((8, 8, NUM_PLANES), dtype=np.float32)
        for step in range(HISTORY):
            base = step * PLANES_PER_STEP
            index = len(self._snapshots) - 1 - step
            if index < 0:  # before the game started: no pieces, "not repeated"
                obs[:, :, base + 12] = 1.0
                continue
            pieces, _, key = self._snapshots[index]
            for square, color, piece_type in pieces:
                rank = chess.square_rank(square) if turn == chess.WHITE else 7 - chess.square_rank(square)
                channel = base + (0 if color == turn else 6) + piece_type - 1  # P, N, B, R, Q, K
                obs[7 - rank, chess.square_file(square), channel] = 1.0
            repeated = self._key_counts[key] - 1  # other occurrences of this position in the game so far
            obs[:, :, base + 12] = float(repeated == 0)
            obs[:, :, base + 13] = float(repeated >= 1)
        b, opp = self.board, not turn
        obs[:, :, 112] = float(turn == chess.BLACK)
        obs[:, :, 113] = self.plies / MAX_PLIES
        obs[:, :, 114] = float(b.has_queenside_castling_rights(turn))
        obs[:, :, 115] = float(b.has_kingside_castling_rights(turn))
        obs[:, :, 116] = float(b.has_queenside_castling_rights(opp))
        obs[:, :, 117] = float(b.has_kingside_castling_rights(opp))
        obs[:, :, 118] = b.halfmove_clock / 100.0
        return obs

    def outcome(self) -> tuple[bool, float]:
        """(game over, value for the side to move): -1 if checkmated, 0 for a draw or a game in progress."""
        b = self.board
        if not any(b.legal_moves):
            return True, -1.0 if b.is_check() else 0.0
        drawn = (b.halfmove_clock >= 100 or b.is_insufficient_material()
                 or self._key_counts[self._snapshots[-1][2]] >= 3 or self.plies >= self.max_plies)
        return (True, 0.0) if drawn else (False, 0.0)
