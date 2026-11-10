"""The pinned Stockfish (decision 2026-11-09: Stockfish 19) as an opponent, over UCI (protocol §8.2).

One thread and HASH_MB of hash, so that a fixed node count gives the same move on any machine. Two kinds of
opponent: full strength at a fixed number of nodes per move, and strength-limited (`elo`, down to MIN_ELO), which
also searches a fixed number of nodes per move. The strength-limited engine picks among its best moves with its own
random numbers, so its games do not repeat; the full-strength engine's do.

The hash table is cleared before every move (decision 2026-11-09), so a full-strength move depends only on the game
so far: not on what the engine searched before, and not on how many games one engine process is serving. One engine
can therefore play many games in turn, and a pool of engines (`EnginePool`) can share them out.

Install the engine with scripts/install_stockfish.py, which checks the release file's checksum and the engine's
bench number.
"""
import queue
from pathlib import Path

import chess
import chess.engine

ROOT = Path(__file__).resolve().parents[2]
TAG = "sf_19"
NAME = "Stockfish 19"  # what the engine must call itself
INSTALL_DIR = ROOT / "build" / "stockfish" / TAG
HASH_MB = 16  # Stockfish's default (decision 2026-11-09): searches of 10,000 nodes or fewer cannot fill it;
# cleared before every move
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

    def new_game(self, game: int | None = None) -> None:
        """Nothing to do: the engine keeps nothing between moves. (`game` seeds the random players.)"""

    def move(self, board: chess.Board) -> chess.Move:
        """The engine's move in `board` (not finished). The engine is sent the game's moves, so it sees repetitions;
        `game=object()` makes python-chess send `ucinewgame` first, which clears the hash table."""
        return self.engine.play(board, self.limit, game=object()).move

    def close(self) -> None:
        self.engine.quit()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


class EnginePool:
    """`size` engines of one kind behind one player: `move` may be called from several threads at once, and each call
    takes whichever engine is free. Since a move depends only on the game so far, it does not matter which."""

    def __init__(self, nodes: int, *, elo: int | None = None, size: int = 1, path: Path | None = None):
        self._engines = [Stockfish(nodes, elo=elo, path=path) for _ in range(size)]
        self.name = self._engines[0].name
        self._free = queue.SimpleQueue()
        for engine in self._engines:
            self._free.put(engine)

    def new_game(self, game: int | None = None) -> None:
        pass

    def move(self, board: chess.Board) -> chess.Move:
        engine = self._free.get()
        try:
            return engine.move(board)
        finally:
            self._free.put(engine)

    def close(self) -> None:
        for engine in self._engines:
            engine.close()


class Analyst:
    """The pinned Stockfish as a judge of positions: full strength, one thread, HASH_MB of hash, searching to a fixed
    depth, each position as a new game. The same position always gets the same answer, on any machine."""

    def __init__(self, depth: int, path: Path | None = None):
        self._engine = Stockfish(1, path=path)  # opens and checks the engine; its node limit is not used here
        self.limit = chess.engine.Limit(depth=depth)

    def centipawns(self, board: chess.Board) -> int | None:
        """Stockfish's evaluation from White's side, in centipawns; None if it sees a forced mate."""
        info = self._engine.engine.analyse(board, self.limit, game=object())
        return info["score"].white().score()

    def close(self) -> None:
        self._engine.close()

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
PINNED_GAME = (  # Stockfish against itself at 100 nodes, 40 plies from the start (hash cleared before every move)
    "d2d4 d7d5 g1f3 c8f5 f3h4 f5e4 f2f3 e7e5 g2g3 e4g6 d4e5 f8e7 h4g6 h7g6 f3f4 g8h6 f1g2 c7c6 e2e4 d8b6 "
    "e4d5 c6d5 b1c3 d5d4 c3d5 e7b4 c2c3 d4c3 b2c3 b4c3 d5c3 e8g8 c3e4 b6a5 d1d2 a5d2 c1d2 f8d8 d2a5 d8d5")


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
