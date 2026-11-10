"""ws_zero.evaluate: our agent (network + search) against an opponent, every opening with both colours."""
import chess
import jax.numpy as jnp
import pytest

from ws_zero.evaluate import play_agent_games
from ws_zero.game.env import ChessEnv
from ws_zero.game.moves import NUM_ACTIONS
from ws_zero.game.reference import ReferenceGame
from ws_zero.players import MaterialGrabber, RandomMover

ENV = ChessEnv()
OPENINGS = [["e2e4", "e7e5", "g1f3", "b8c6"], ["d2d4", "d7d5", "c2c4", "e7e6"]]
PIECES = jnp.array([1.0, 3.0, 3.0, 5.0, 9.0, 0.0])


def _material(params, obs):
    """Every move equally likely; value = tanh(material difference / 4) for the side to move (the Phase 4 dummy)."""
    diff = jnp.einsum("bhwp,p->b", obs[..., 0:6], PIECES) - jnp.einsum("bhwp,p->b", obs[..., 6:12], PIECES)
    return jnp.zeros((obs.shape[0], NUM_ACTIONS)), jnp.tanh(diff / 4)


def _first_legal(params, obs):
    """A policy with a clear top move: logits fall with the action number."""
    return jnp.broadcast_to(-jnp.arange(NUM_ACTIONS, dtype=jnp.float32), (obs.shape[0], NUM_ACTIONS)), jnp.zeros(obs.shape[0])


@pytest.fixture(scope="module")
def games():
    return play_agent_games(ENV, _material, None, 16, lambda game: RandomMover(seed=0), OPENINGS, agent_name="material")


def test_every_opening_is_played_once_with_each_colour(games):
    assert [(g.white, g.black) for g in games] == [("material", "Random mover"), ("Random mover", "material")] * 2
    assert [g.moves[:4] for g in games] == [OPENINGS[0], OPENINGS[0], OPENINGS[1], OPENINGS[1]]


def test_the_games_are_legal_and_end_as_the_project_rules_say(games):
    for g in games:
        ref = ReferenceGame(repetition_key="pgx")
        for uci in g.moves:
            assert ref.end_reason() is None  # no move after the game is over
            move = chess.Move.from_uci(uci)
            assert move in ref.board.legal_moves
            ref.push(move)
        assert ref.end_reason() == g.end
        expected = "1/2-1/2" if g.end != "checkmate" else ("0-1" if ref.board.turn == chess.WHITE else "1-0")
        assert g.result == expected


def test_the_agents_values_are_recorded_at_its_own_moves_only(games):
    for g in games:
        agent_is_white = g.white == "material"
        for ply, value in enumerate(g.values):
            agents_move = ply >= 4 and (ply % 2 == 0) == agent_is_white
            assert (value is not None) == agents_move
            assert value is None or -1 <= value <= 1


def test_the_same_seed_replays_the_same_games_and_another_seed_differs(games):
    again = play_agent_games(ENV, _material, None, 16, lambda game: RandomMover(seed=0), OPENINGS, agent_name="material")
    assert [g.moves for g in again] == [g.moves for g in games]
    other = play_agent_games(ENV, _material, None, 16, lambda game: RandomMover(seed=1), OPENINGS, agent_name="material")
    assert [g.moves for g in other] != [g.moves for g in games]


def test_each_game_gets_its_own_opponent_made_with_its_game_number():
    made = []

    def make(game):
        made.append(game)
        return MaterialGrabber(seed=0)
    play_agent_games(ENV, _first_legal, None, 0, make, OPENINGS[:1], agent_name="policy")
    assert made == [0, 1]


def test_with_no_search_the_agent_plays_its_policys_top_legal_move():
    games = play_agent_games(ENV, _first_legal, None, 0, lambda game: RandomMover(seed=0), OPENINGS[:1],
                             agent_name="policy")
    game = games[0]  # the agent is White
    ref = ReferenceGame(repetition_key="pgx")
    for ply, uci in enumerate(game.moves):
        if ply >= 4 and ply % 2 == 0:
            assert ref.legal_actions()[0] == __import__("ws_zero.game.moves", fromlist=["x"]).move_to_action(
                ref.board, chess.Move.from_uci(uci))
        ref.push(chess.Move.from_uci(uci))
