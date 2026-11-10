"""Evaluation games (protocol §8.3): our agent, a network with search, against one opponent.

Every opening is played twice, the agent once with each colour, all games side by side: the agent's moves for every
game come from one batched search on the GPU, and each game has its own opponent object, so no game's moves depend
on another's. The agent plays as in the protocol: no root noise, the most-visited move from its first move; with 0
simulations it plays its policy's top legal move instead (the secondary agent).

The positions are kept twice, in pgx for the search and in python-chess (`ReferenceGame`, with pgx's repetition
rule) for the opponents and the records. The two must agree after every move on whether the game is over; if they
ever do not, the run stops.
"""
from concurrent.futures import ThreadPoolExecutor

import chess
import jax
import jax.numpy as jnp
import numpy as np

from ws_zero.game.moves import action_to_move, move_to_action
from ws_zero.game.reference import ReferenceGame
from ws_zero.net import ILLEGAL_LOGIT
from ws_zero.players import Game
from ws_zero.search import search_moves


def make_agent(env, evaluate, params, simulations: int):
    """(state, key) -> (one action per game, the agent's value estimate for the side to move). Build it once and
    pass it to `play_agent_games(agent=...)` for every opponent, so that it is compiled once."""
    if simulations == 0:
        def policy(state, key):
            logits, value = evaluate(params, env.observe(state))
            return jnp.argmax(jnp.where(state.legal_action_mask, logits, ILLEGAL_LOGIT), axis=-1), value
        return jax.jit(policy)

    def search(state, key):
        actions, out = search_moves(env, evaluate, params, key, state, simulations, self_play=False)
        return actions, out.search_tree.summary().value
    return jax.jit(search)


def _states(env, action_lists):
    """The pgx states reached by each game's list of actions (lists may differ in length), in the games' order."""
    order, parts = [], []
    for length in sorted({len(a) for a in action_lists}):
        group = [g for g, a in enumerate(action_lists) if len(a) == length]
        state = env.init(len(group))
        for t in range(length):
            state = env.step(state, [action_lists[g][t] for g in group])
        order += group
        parts.append(state)
    joined = jax.tree.map(lambda *xs: jnp.concatenate(xs), *parts)
    back = np.argsort(np.asarray(order))
    return jax.tree.map(lambda x: x[back], joined)


def play_agent_games(env, evaluate, params, simulations: int, make_opponent, openings, *, agent_name: str = "agent",
                     seed: int = 0, threads: int = 1, agent=None) -> list[Game]:
    """Play 2 x len(openings) games: game 2i is opening i with the agent as White, game 2i + 1 with the agent as
    Black. `make_opponent(game number)` returns that game's opponent (a player, see ws_zero.players). `openings` are
    lists of UCI moves, all of one length. `seed` seeds the search's tie-breaking. With `threads` above 1 the
    opponents' moves of one turn are computed side by side (each game's opponent is only ever used by one thread at a
    time; opponents that share engines must share them through an `EnginePool`)."""
    total = 2 * len(openings)
    refs = [ReferenceGame(repetition_key="pgx") for _ in range(total)]
    agent_is_white = [g % 2 == 0 for g in range(total)]
    opponents, actions, values = [], [[] for _ in range(total)], [[] for _ in range(total)]
    for g in range(total):
        opponent = make_opponent(g)
        opponent.new_game(g)
        opponents.append(opponent)

    def push(g, move, value=None):
        if move not in refs[g].board.legal_moves:
            raise ValueError(f"illegal move {move} in game {g}: {refs[g].board.fen()}")
        action = move_to_action(refs[g].board, move)
        refs[g].push(move)
        values[g].append(value)
        return action

    def agent_to_move(g):
        return (refs[g].board.turn == chess.WHITE) == agent_is_white[g]

    pool = ThreadPoolExecutor(threads) if threads > 1 else None

    def opponents_move(live):
        ask = lambda g: opponents[g].move(refs[g].board.copy())  # noqa: E731
        moves = list(pool.map(ask, live)) if pool else [ask(g) for g in live]
        step = np.zeros(total, dtype=np.int32)
        for g, move in zip(live, moves):
            step[g] = push(g, move)
        return step

    for g in range(total):  # the opening, then the opponent's first move in the games where it moves first
        for uci in openings[g // 2]:
            actions[g].append(push(g, chess.Move.from_uci(uci)))
    first = [g for g in range(total) if refs[g].end_reason() is None and not agent_to_move(g)]
    for g, action in zip(first, opponents_move(first)[first]):
        actions[g].append(int(action))
    state = _states(env, actions)

    def live_games(state):
        over = np.asarray(env.is_terminal(state))
        ours = np.array([ref.end_reason() is not None for ref in refs])
        if (over != ours).any():
            g = int(np.flatnonzero(over != ours)[0])
            raise RuntimeError(f"pgx and the reference disagree on whether game {g} is over: {refs[g].board.fen()}")
        return [int(g) for g in np.flatnonzero(~over)]

    agent, key = agent or make_agent(env, evaluate, params, simulations), jax.random.key(seed)
    live = live_games(state)
    while live:
        key, sub = jax.random.split(key)
        chosen, estimate = (np.asarray(x) for x in agent(state, sub))
        step = np.zeros(total, dtype=np.int32)
        for g in live:
            assert agent_to_move(g)
            step[g] = push(g, action_to_move(refs[g].board, int(chosen[g])), round(float(estimate[g]), 3))
        state = env.step(state, step)
        live = live_games(state)
        if live:
            state = env.step(state, opponents_move(live))
            live = live_games(state)

    if pool:
        pool.shutdown()
    games = []
    for g, ref in enumerate(refs):
        end = ref.end_reason()
        result = "1/2-1/2" if end != "checkmate" else ("0-1" if ref.board.turn == chess.WHITE else "1-0")
        white, black = (agent_name, opponents[g].name) if agent_is_white[g] else (opponents[g].name, agent_name)
        games.append(Game(white, black, None, [m.uci() for m in ref.board.move_stack], result, end, values[g]))
    return games
