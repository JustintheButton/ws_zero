"""The weak anchors (protocol §8.2) and the game loop that plays any two players."""
from collections import Counter

import chess
import pytest

from ws_zero.players import MaterialGrabber, RandomMover, play_game


class Scripted:
    """Test player: plays a fixed list of moves, in order."""
    name = "scripted"

    def __init__(self, moves):
        self.moves = list(moves)

    def new_game(self, game=0):
        self.todo = iter(self.moves)

    def move(self, board):
        return chess.Move.from_uci(next(self.todo))


def _moves(player, fen, games):
    out = []
    for game in range(games):
        player.new_game(game)
        out.append(player.move(chess.Board(fen)).uci())
    return out


# ---------- random mover ----------

def test_the_random_mover_plays_every_legal_move_about_equally_often():
    counts = Counter(_moves(RandomMover(seed=0), chess.STARTING_FEN, 2000))
    assert len(counts) == 20  # the start position has 20 legal moves
    assert min(counts.values()) > 60 and max(counts.values()) < 140  # 100 expected each; 4 standard deviations


def test_the_random_mover_repeats_a_game_given_its_seed_and_game_number():
    a, b = RandomMover(seed=7), RandomMover(seed=7)
    assert _moves(a, chess.STARTING_FEN, 50) == _moves(b, chess.STARTING_FEN, 50)
    assert _moves(a, chess.STARTING_FEN, 50) != _moves(RandomMover(seed=8), chess.STARTING_FEN, 50)


# ---------- greedy material-grabber ----------

def test_the_grabber_takes_the_most_valuable_piece_on_offer():
    fen = "4k3/8/8/2r1q3/3P4/8/8/4K3 w - - 0 1"  # the d4 pawn can take a rook (c5) or a queen (e5)
    assert set(_moves(MaterialGrabber(seed=0), fen, 30)) == {"d4e5"}


def test_the_grabber_counts_en_passant_as_taking_a_pawn():
    fen = "4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1"  # the only capture is e5xd6 en passant
    assert set(_moves(MaterialGrabber(seed=0), fen, 30)) == {"e5d6"}


def test_the_grabber_picks_at_random_between_captures_of_equal_value():
    fen = "4k3/3n4/8/8/b2R4/8/8/4K3 w - - 0 1"  # the rook can take a knight (d7) or a bishop (a4), both worth 3
    counts = Counter(_moves(MaterialGrabber(seed=0), fen, 200))
    assert set(counts) == {"d4d7", "d4a4"}
    assert min(counts.values()) > 60


def test_the_grabber_plays_a_random_move_when_nothing_can_be_taken():
    counts = Counter(_moves(MaterialGrabber(seed=0), chess.STARTING_FEN, 2000))
    assert len(counts) == 20 and min(counts.values()) > 60 and max(counts.values()) < 140


# ---------- the game loop ----------

def test_a_game_ends_at_checkmate_with_the_winner_recorded():
    game = play_game(Scripted(["g2g4"]), Scripted(["d8h4"]), opening=["f2f3", "e7e5"])  # fool's mate
    assert (game.result, game.end) == ("0-1", "checkmate")
    assert game.moves == ["f2f3", "e7e5", "g2g4", "d8h4"]


def test_a_game_is_drawn_when_a_position_comes_up_three_times():
    white, black = Scripted(["g1f3", "f3g1"] * 2), Scripted(["g8f6", "f6g8"] * 2)
    game = play_game(white, black)
    assert (game.result, game.end, len(game.moves)) == ("1/2-1/2", "threefold repetition", 8)


def test_a_game_is_drawn_at_the_ply_cap_and_the_opening_counts_towards_it():
    game = play_game(RandomMover(seed=0), RandomMover(seed=1), opening=["e2e4", "e7e5"], max_plies=10)
    assert (game.result, game.end, len(game.moves)) == ("1/2-1/2", "ply cap", 10)


def test_stalemate_is_a_draw():
    # Black to move after Qf7 has no legal move and is not in check.
    game = play_game(Scripted(["f6f7"]), Scripted([]), start_fen="7k/8/5QK1/8/8/8/8/8 w - - 0 1")
    assert (game.result, game.end) == ("1/2-1/2", "stalemate")


def test_the_same_players_and_game_number_replay_the_same_game():
    def game(number):
        return play_game(RandomMover(seed=0), MaterialGrabber(seed=1), game=number).moves
    assert game(3) == game(3)
    assert game(3) != game(4)


def test_an_illegal_move_stops_the_game_with_an_error():
    with pytest.raises(ValueError, match="illegal"):
        play_game(Scripted(["e2e5"]), Scripted([]))


# ---------- diluted opponents: a base player that plays a random move some share of the time ----------

class Fixed:
    """Test base player: always plays e2e4 (legal in the start position) and counts its games and moves."""
    name = "fixed"

    def __init__(self):
        self.games, self.calls = [], 0

    def new_game(self, game=0):
        self.games.append(game)

    def move(self, board):
        self.calls += 1
        return chess.Move.from_uci("e2e4")


def test_a_fully_diluted_player_never_asks_its_base_and_an_undiluted_one_always_does():
    from ws_zero.players import Diluted
    base = Fixed()
    assert len(set(_moves(Diluted(base, 1.0, seed=0), chess.STARTING_FEN, 200))) == 20 and base.calls == 0
    base = Fixed()
    assert set(_moves(Diluted(base, 0.0, seed=0), chess.STARTING_FEN, 200)) == {"e2e4"} and base.calls == 200


def test_a_half_diluted_player_asks_its_base_about_half_the_time():
    from ws_zero.players import Diluted
    base = Fixed()
    _moves(Diluted(base, 0.5, seed=0), chess.STARTING_FEN, 2000)
    assert 910 < base.calls < 1090  # 1,000 expected; 4 standard deviations


def test_a_diluted_player_replays_given_its_seed_and_tells_its_base_about_each_game():
    from ws_zero.players import Diluted
    base = Fixed()
    player = Diluted(base, 0.5, seed=3)
    first = _moves(player, chess.STARTING_FEN, 60)
    assert first == _moves(Diluted(Fixed(), 0.5, seed=3), chess.STARTING_FEN, 60)
    assert first != _moves(Diluted(Fixed(), 0.5, seed=4), chess.STARTING_FEN, 60)
    assert base.games == list(range(60))
    assert player.name == "fixed, 50% random moves"
