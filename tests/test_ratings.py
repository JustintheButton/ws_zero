"""ws_zero.ratings: one Elo rating per player from all the pairings together, with ranges from resampling."""
import numpy as np
import pytest

from ws_zero.ratings import fit_ratings, resampled_ratings


def _pairing(low, high, wins, draws, losses):
    games = wins + draws + losses
    return {"low": low, "high": high, "games": games, "score": (wins + draws / 2) / games,
            "wins": wins, "draws": draws, "losses": losses}


def test_ratings_put_a_75_percent_score_191_elo_apart_and_chain_through_a_middle_player():
    # b scores 75% against a, c scores 75% against b; a and c never meet. 75% is 400 x log10(3) = 190.8 Elo.
    pairings = [_pairing("a", "b", 150, 0, 50), _pairing("b", "c", 150, 0, 50)]
    ratings = fit_ratings(["a", "b", "c"], pairings)
    assert ratings["a"] == 0
    assert ratings["b"] == pytest.approx(190.8, abs=0.5)
    assert ratings["c"] == pytest.approx(381.7, abs=1.0)


def test_resampled_ratings_spread_around_the_fit_and_narrow_with_more_games():
    names = ["a", "b"]
    few = resampled_ratings(names, [_pairing("a", "b", 30, 0, 10)], samples=300, seed=0)["b"]
    many = resampled_ratings(names, [_pairing("a", "b", 3000, 0, 1000)], samples=300, seed=0)["b"]
    assert np.percentile(few, 2.5) < 190.8 < np.percentile(few, 97.5)
    assert np.percentile(many, 2.5) < 190.8 < np.percentile(many, 97.5)
    assert np.std(many) < np.std(few) / 5  # 100 times the games: about a tenth of the spread


def test_two_players_with_the_same_results_are_not_told_apart():
    names = ["anchor", "x", "y"]
    pairings = [_pairing("anchor", "x", 60, 20, 20), _pairing("anchor", "y", 60, 20, 20)]
    samples = resampled_ratings(names, pairings, samples=400, seed=1)
    difference = samples["x"] - samples["y"]
    assert np.percentile(difference, 2.5) < 0 < np.percentile(difference, 97.5)


def test_a_player_that_loses_every_game_still_gets_a_finite_rating():
    ratings = fit_ratings(["a", "b"], [_pairing("a", "b", 100, 0, 0)])
    assert np.isfinite(ratings["b"]) and ratings["b"] > 800
