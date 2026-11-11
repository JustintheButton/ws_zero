"""One Elo rating per player from many pairings together, with ranges from resampling the games.

The model is Bradley-Terry: the chance that i beats j is s_i / (s_i + s_j), a draw counting as half a win and half a
loss; a rating is 400 x log10(s). The first player named is fixed at 0. A stand-in for Ordo (protocol §8.4), to be
cross-checked against it when the real evaluation starts.

A pairing is {"low", "high", "games", "score"} where score is `high`'s share of the points; resampling also needs
"wins", "draws" and "losses" (`high`'s).
"""
import numpy as np

PRIOR_GAMES = 0.01  # a hundredth of a drawn game added to every pairing played, so a 100% score gives a finite rating


def _tables(names, pairings):
    index = {name: i for i, name in enumerate(names)}
    points, games = np.zeros((len(names), len(names))), np.zeros((len(names), len(names)))
    for p in pairings:
        low, high = index[p["low"]], index[p["high"]]
        points[high, low] += p["score"] * p["games"] + PRIOR_GAMES / 2
        points[low, high] += (1 - p["score"]) * p["games"] + PRIOR_GAMES / 2
        games[high, low] += p["games"] + PRIOR_GAMES
        games[low, high] += p["games"] + PRIOR_GAMES
    return points, games


def _fit(points, games, strength=None, steps=20000):
    strength = np.ones(len(points)) if strength is None else strength.copy()
    for _ in range(steps):
        strength = points.sum(axis=1) / (games / (strength[:, None] + strength[None, :])).sum(axis=1)
        strength /= strength[0]
    return strength


def fit_ratings(names: list[str], pairings: list[dict]) -> dict[str, float]:
    """Each player's rating, the first player at 0."""
    strength = _fit(*_tables(names, pairings))
    return {name: float(400 * np.log10(strength[i])) for i, name in enumerate(names)}


def resampled_ratings(names: list[str], pairings: list[dict], samples: int = 500, seed: int = 0) -> dict:
    """`samples` refits, each on games redrawn with replacement within every pairing: name -> array of ratings.
    Percentiles of an array give a player's range; percentiles of a difference of two arrays, the range of a gap."""
    rng = np.random.default_rng(seed)
    start = _fit(*_tables(names, pairings))
    out = np.zeros((samples, len(names)))
    for s in range(samples):
        redrawn = []
        for p in pairings:
            wins, draws, _ = rng.multinomial(p["games"], [p["wins"] / p["games"], p["draws"] / p["games"],
                                                          p["losses"] / p["games"]])
            redrawn.append({**p, "score": (wins + draws / 2) / p["games"]})
        out[s] = 400 * np.log10(_fit(*_tables(names, redrawn), strength=start, steps=3000))
    return {name: out[:, i] for i, name in enumerate(names)}
