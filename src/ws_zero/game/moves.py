"""The 4,672-action move encoding (AlphaZero's 8 x 8 x 73), in pgx's exact layout, mapped to python-chess moves.

action = from_index * 73 + plane
- from_index = file * 8 + rank (pgx numbers squares file by file: a1 = 0, a2 = 1, ..., h8 = 63), seen from the side
  to move: ranks are mirrored when Black is to move (rank -> 7 - rank); files never change.
- planes 0-8: underpromotions. plane // 3 is the piece (0 rook, 1 bishop, 2 knight); plane % 3 is the direction
  (0 straight ahead, 1 capture toward the h-file, 2 capture toward the a-file).
- planes 9-72: every other move, as a (rank, file) displacement from `_DELTAS` (56 queen-style moves, then 8
  knight moves). A pawn reaching the last rank by one of these planes promotes to a queen. Castling is the king
  moving two squares.
"""
import chess

NUM_ACTIONS = 64 * 73

_UNDERPROMOTION_PIECES = (chess.ROOK, chess.BISHOP, chess.KNIGHT)
_UNDERPROMOTION_DFILE = (0, +1, -1)


def _queen_and_knight_deltas() -> list[tuple[int, int]]:
    """(rank change, file change) for planes 9..72, in pgx's order."""
    up, down, none = list(range(1, 8)), list(range(-7, 0)), [0] * 7
    d_rank = down + up + none + none + down + up + up[::-1] + down[::-1] + [-1, +1, -2, +2, -1, +1, -2, +2]
    d_file = none + none + down + up + down + up + down + up + [-2, -2, -1, -1, +2, +2, +1, +1]
    return list(zip(d_rank, d_file))


_DELTAS = _queen_and_knight_deltas()
_PLANE_OF_DELTA = {delta: 9 + i for i, delta in enumerate(_DELTAS)}


def _relative(square: int, turn: chess.Color) -> tuple[int, int]:
    rank = chess.square_rank(square)
    return chess.square_file(square), rank if turn == chess.WHITE else 7 - rank


def move_to_action(board: chess.Board, move: chess.Move) -> int:
    """The action index of a legal `move` in `board`."""
    f0, r0 = _relative(move.from_square, board.turn)
    f1, r1 = _relative(move.to_square, board.turn)
    if move.promotion in _UNDERPROMOTION_PIECES:
        plane = 3 * _UNDERPROMOTION_PIECES.index(move.promotion) + _UNDERPROMOTION_DFILE.index(f1 - f0)
    else:
        plane = _PLANE_OF_DELTA[(r1 - r0, f1 - f0)]
    return (f0 * 8 + r0) * 73 + plane


def action_to_move(board: chess.Board, action: int) -> chess.Move:
    """The python-chess move for `action` in `board` (raises ValueError if the action leaves the board)."""
    from_index, plane = divmod(int(action), 73)
    f0, r0 = divmod(from_index, 8)
    promotion = None
    if plane < 9:
        promotion = _UNDERPROMOTION_PIECES[plane // 3]
        f1, r1 = f0 + _UNDERPROMOTION_DFILE[plane % 3], r0 + 1
    else:
        d_rank, d_file = _DELTAS[plane - 9]
        f1, r1 = f0 + d_file, r0 + d_rank
    if not (0 <= f1 < 8 and 0 <= r1 < 8):
        raise ValueError(f"action {action} moves off the board")
    white = board.turn == chess.WHITE
    from_square = chess.square(f0, r0 if white else 7 - r0)
    to_square = chess.square(f1, r1 if white else 7 - r1)
    if promotion is None and r1 == 7 and board.piece_type_at(from_square) == chess.PAWN:
        promotion = chess.QUEEN
    return chess.Move(from_square, to_square, promotion)
