"""The evaluation ladder (protocol §8.2, as amended 2026-11-09): its twelve opponents by short name, weakest first.

`opponent(name)` returns (make, close): `make(game number)` gives the player for one game, and `close()` shuts down
any engines. Players that need Stockfish share a pool of `engines` engine processes.
"""
from ws_zero.players import Diluted, MaterialGrabber, RandomMover
from ws_zero.stockfish import DILUTED_RANDOM_SHARES, LADDER_NODES, LIMITED_NODES, MIN_ELO, EnginePool

DILUTED = {f"sf1r{round(100 * share)}": share for share in DILUTED_RANDOM_SHARES}
LADDER = ["random", "grabber", *DILUTED, "sf1320", *(f"sf{n}" for n in LADDER_NODES)]


def opponent(name: str, engines: int = 1):
    if name == "random":
        return (lambda game: RandomMover(seed=0)), (lambda: None)
    if name == "grabber":
        return (lambda game: MaterialGrabber(seed=1)), (lambda: None)
    if name in DILUTED:
        pool = EnginePool(1, size=engines)
        return (lambda game: Diluted(pool, DILUTED[name], seed=2)), pool.close
    if name == "sf1320":
        pool = EnginePool(LIMITED_NODES, elo=MIN_ELO, size=engines)
        return (lambda game: pool), pool.close
    if name in LADDER:
        pool = EnginePool(int(name[2:]), size=engines)
        return (lambda game: pool), pool.close
    raise ValueError(f"unknown opponent {name!r}; the ladder is {', '.join(LADDER)}")
