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
