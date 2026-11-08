# Encodings

The exact input and output encodings, as implemented by pgx 2.6.0 (`ws_zero.game.env`) and reproduced
independently on python-chess (`ws_zero.game.reference`, `ws_zero.game.moves`). The two agree on 680,319 positions
from 2,000 random games ([gate report](gates/phase-2.md)). They follow AlphaZero (Silver et al., 2018); where the
paper leaves a choice open, pgx's choice is documented here. The WSP prompt is built from this file.

## Observation: 8 × 8 × 119, from the side to move's perspective

**Orientation.** Row 0 is the far rank (the opponent's back rank at the start), row 7 is the side to move's own
back rank. Column 0 is the a-file, column 7 the h-file. When Black is to move the board is mirrored top to bottom
only: files are never mirrored. So for White to move, row r is rank 8 − r; for Black to move, row r is rank r + 1.

**"Mine" and "theirs"** always refer to the player to move *now*, in every history step.

### Planes 0–111: eight history steps × 14 planes

Step t = 0 is the current position, t = 1 the position one ply earlier, …, t = 7. Step t uses planes 14t … 14t + 13:

| Offset | Meaning (1 on the squares where it holds, else 0) |
|---|---|
| 0–5 | my pawns, knights, bishops, rooks, queens, king |
| 6–11 | their pawns, knights, bishops, rooks, queens, king |
| 12 | 1 everywhere if this position occurs **nowhere else** in the game so far |
| 13 | 1 everywhere if this position occurs **at least once elsewhere** in the game so far |

- History steps before the game began (and before the starting position of a game set up from FEN): every piece
  plane is 0, offset 12 is 1, offset 13 is 0.
- "Elsewhere in the game so far" counts every position from the game's start to the current position, including
  positions later than step t.
- Two positions are the same if the pieces, the side to move, the castling rights and the en passant square are
  the same. pgx sets the en passant square after **every** two-square pawn move, even when no capture is possible;
  FIDE counts it only when a capture is legal. The two differ in about 0.007% of positions
  ([decisions.md](decisions.md)).

### Planes 112–118: constant planes (the same value on every square)

| Plane | Value |
|---|---|
| 112 | 1 if Black is to move, 0 if White |
| 113 | plies played since the game's start ÷ 512 |
| 114 | 1 if I may still castle queen-side (the right, not whether it is legal now) |
| 115 | 1 if I may still castle king-side |
| 116 | 1 if they may still castle queen-side |
| 117 | 1 if they may still castle king-side |
| 118 | plies since the last capture or pawn move ÷ 100 |

Planes 113 and 118 are computed by XLA and may differ from exact division in the last bit.

## Actions: 4,672 = 64 squares × 73 move types

`action = from_index × 73 + plane`

**From-square index.** `from_index = file × 8 + rank`, files a–h = 0–7, and ranks counted from the side to move's
own back rank (0) to the far rank (7). Squares are numbered file by file: a1 = 0, a2 = 1, …, a8 = 7, b1 = 8, …,
h8 = 63 for White. For Black, ranks are mirrored: Black's e7 pawn is file 4, rank 1, index 33. Note that this
differs from the observation's row order: square (row, column) of the observation has
`from_index = column × 8 + (7 − row)`.

**Planes 0–8: underpromotions** (from the seventh rank only). Piece = [rook, bishop, knight][plane ÷ 3];
direction = [straight ahead, capture toward the h-file, capture toward the a-file][plane mod 3].

**Planes 9–72: all other moves**, as a displacement (Δrank, Δfile) from the side to move's perspective
(forward = +Δrank, toward the h-file = +Δfile):

| Planes | Displacements, in plane order |
|---|---|
| 9–15 | backward: (−7, 0), (−6, 0), …, (−1, 0) |
| 16–22 | forward: (+1, 0), …, (+7, 0) |
| 23–29 | toward the a-file: (0, −7), …, (0, −1) |
| 30–36 | toward the h-file: (0, +1), …, (0, +7) |
| 37–43 | backward, toward a: (−7, −7), …, (−1, −1) |
| 44–50 | forward, toward h: (+1, +1), …, (+7, +7) |
| 51–57 | forward, toward a: (+7, −7), …, (+1, −1) |
| 58–64 | backward, toward h: (−1, +1), …, (−7, +7) |
| 65–72 | knight: (−1, −2), (+1, −2), (−2, −1), (+2, −1), (−1, +2), (+1, +2), (−2, +1), (+2, +1) |

- A pawn reaching the far rank through planes 9–72 promotes to a **queen**.
- **Castling** is the king moving two squares: king-side = e1 (index 32), plane 31 → action 2,367; queen-side =
  plane 28 → action 2,364 (the same numbers for Black, by mirroring).
- Illegal actions are masked out before the softmax; the legal-action mask comes from pgx.

## Outcomes

A game ends at checkmate, stalemate, insufficient material (king vs king; king and one minor piece vs king; kings
and bishops all on squares of one color), threefold repetition (same identity as above), 100 plies without a
capture or pawn move, or 512 plies. The value of a terminal position for the side to move is −1 if checkmated,
else 0.
