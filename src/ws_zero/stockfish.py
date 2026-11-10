"""The pinned Stockfish (decision 2026-11-09: Stockfish 19) as an opponent, over UCI (protocol §8.2).

One thread and HASH_MB of hash, so that a fixed node count gives the same move on any machine. Two kinds of
opponent: full strength at a fixed number of nodes per move, and strength-limited (`elo`, down to MIN_ELO), which
also searches a fixed number of nodes per move. The strength-limited engine picks among its best moves with its own
random numbers, so its games do not repeat; the full-strength engine's do.

The hash table is kept between the moves of one game, as in normal play, and cleared at the start of each game. A
new game starts by itself whenever the position asked about does not continue the previous one (the same starting
position with more moves played), so a game replays move for move whatever was played before it, and a single
position always gives the same move.

Install the engine with scripts/install_stockfish.py, which checks the release file's checksum and the engine's
bench number.
"""
from pathlib import Path

import chess
import chess.engine

ROOT = Path(__file__).resolve().parents[2]
TAG = "sf_19"
NAME = "Stockfish 19"  # what the engine must call itself
INSTALL_DIR = ROOT / "build" / "stockfish" / TAG
HASH_MB = 16  # Stockfish's default (decision 2026-11-09): searches of 10,000 nodes or fewer cannot fill it
MIN_ELO = 1320  # the lowest UCI_Elo Stockfish 19 accepts
LADDER_NODES = (1, 100, 300, 1000, 3000, 10000)  # protocol §8.2 (amended 2026-11-09); 1 = a one-move-deep search
LIMITED_NODES = 10000  # nodes per move of the strength-limited opponent (decision 2026-11-09)
DILUTED_RANDOM_SHARES = (0.75, 0.50, 0.25)  # the diluted rungs: the 1-node engine with this share of random moves


def find_engine(folder: Path) -> Path | None:
    """The engine binary among the unpacked release files: the largest file whose name starts with "stockfish"."""
    files = [p for p in Path(folder).rglob("stockfish*") if p.is_file() and not p.name.endswith((".tar.gz", ".zip"))]
    return max(files, key=lambda p: p.stat().st_size) if files else None


def installed_engine() -> Path | None:
    """The engine installed by scripts/install_stockfish.py, or None."""
    return find_engine(INSTALL_DIR) if INSTALL_DIR.exists() else None


class Stockfish:
    """`Stockfish(nodes)`: full strength at `nodes` per move. `Stockfish(nodes, elo=1320)`: strength-limited."""

    def __init__(self, nodes: int, *, elo: int | None = None, path: Path | None = None):
        path = path or installed_engine()
        if path is None:
            raise RuntimeError("Stockfish is not installed: run scripts/install_stockfish.py")
        self.name = f"Stockfish, {nodes:,} nodes" + (f", limited to {elo}" if elo is not None else "")
        self.limit = chess.engine.Limit(nodes=nodes)
        self.engine = chess.engine.SimpleEngine.popen_uci(str(path))
        name = self.engine.id.get("name")
        if name != NAME:
            self.engine.quit()
            raise RuntimeError(f"{path} calls itself {name!r}, not {NAME!r}: not the project's engine")
        options = {"Threads": 1, "Hash": HASH_MB}
        if elo is not None:
            options.update(UCI_LimitStrength=True, UCI_Elo=elo)
        self.engine.configure(options)
        self.new_game()

    def new_game(self, game: int | None = None) -> None:
        """Start a new game: the engine clears its hash table before its next move. (`game`, the game's number, is
        what the random players are seeded with; Stockfish does not use it.)"""
        self._game = object()  # python-chess sends `ucinewgame` when this changes
        self._seen = None

    def move(self, board: chess.Board) -> chess.Move:
        """The engine's move in `board` (not finished). The engine is sent the game's moves, so it sees repetitions."""
        line = (board.root().fen(), tuple(board.move_stack))
        continues = (self._seen is not None and line[0] == self._seen[0] and len(line[1]) > len(self._seen[1])
                     and line[1][:len(self._seen[1])] == self._seen[1])
        if not continues:
            self.new_game()
        self._seen = line
        return self.engine.play(board, self.limit, game=self._game).move

    def close(self) -> None:
        self.engine.quit()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


# What the Mac's engine played on 2026-11-09. Stockfish at one thread and a fixed node count is the same on every
# machine, so every machine must reproduce these exactly; a difference means a different engine.
PINNED_POSITIONS = [
    chess.STARTING_FEN,
    "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3",  # Italian
    "r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4",  # Qxf7# is there
    "8/5k2/8/3p4/3P4/2K5/8/8 w - - 0 1",  # pawn ending
]
PINNED_MOVES = {  # nodes -> the move in each of PINNED_POSITIONS
    1: ["d2d4", "g8f6", "h5f7", "c3b4"],
    10: ["d2d4", "g8f6", "h5f7", "c3b4"],
    100: ["d2d4", "g8f6", "h5f7", "c3d2"],
    1000: ["e2e4", "g8f6", "h5f7", "c3c2"],
    10000: ["e2e4", "g8f6", "h5f7", "c3b4"],
}
PINNED_GAME = (  # Stockfish against itself at 100 nodes, 40 plies from the start
    "d2d4 d7d5 g1f3 g8f6 c2c3 c8f5 d1b3 d8c8 c1f4 b8d7 b1d2 a7a5 e2e3 e7e6 a2a4 f6h5 f4e5 f8e7 f1b5 c7c6 "
    "b5e2 h5f6 e5f4 h7h6 h2h3 e8g8 f3h4 f5h7 h4f3 g8h8 e1g1 f8g8 f3e5 d7e5 f4e5 f6d7 e5g3 g8f8 c3c4 e7b4")


def check_pinned_moves(path: Path | None = None) -> None:
    """Raise unless this machine's engine plays the pinned moves and the pinned game."""
    for nodes, expected in PINNED_MOVES.items():
        with Stockfish(nodes, path=path) as engine:
            played = [engine.move(chess.Board(fen)).uci() for fen in PINNED_POSITIONS]
        if played != expected:
            raise RuntimeError(f"at {nodes} nodes this engine plays {played}, not the pinned {expected}")
    board = chess.Board()
    with Stockfish(100, path=path) as engine:
        while len(board.move_stack) < 40:
            board.push(engine.move(board))
    played = " ".join(m.uci() for m in board.move_stack)
    if played != PINNED_GAME:
        raise RuntimeError(f"this engine's 100-node game differs from the pinned one: {played}")
