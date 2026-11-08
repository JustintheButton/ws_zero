"""Batched chess on pgx (JAX): many games stepped in parallel, on CPU or GPU.

A batch is a pgx `GameState` whose arrays have a leading batch axis. Encodings follow AlphaZero (docs/encoding.md):
`observe` gives (batch, 8, 8, 119) and actions are indices into 4,672 (ws_zero.game.moves).

pgx is used through its low-level `Game` so that colors are explicit (no random player order).

Fix for pgx 2.6 under JAX >= 0.5: pgx builds its Zobrist hash tables from JAX's random generator at import time but
hard-codes the start position's hash computed under an older generator, so repetitions of the start position were
never detected. `initial_state` stores the correctly computed hash instead.
"""
import jax
import jax.numpy as jnp
from pgx._src.games.chess import _zobrist_hash
from pgx.experimental.chess import Game, GameState, State, from_fen, to_fen


def initial_state() -> GameState:
    """The start position, with a hash history consistent with pgx's hash function."""
    state = GameState()
    history = jnp.zeros_like(state.hash_history).at[0].set(_zobrist_hash(state))
    return state._replace(hash_history=history)


def state_from_fen(fen: str) -> GameState:
    """A single position (no move history) from FEN."""
    return from_fen(fen)._x


def state_to_fen(state: GameState) -> str:
    """FEN of a single (unbatched) state."""
    return to_fen(State(_x=state))


class ChessEnv:
    def __init__(self):
        game = Game()
        self._initial = initial_state()

        def step(state: GameState, actions: jax.Array) -> GameState:
            done = jax.vmap(game.is_terminal)(state)
            stepped = jax.vmap(game.step)(state, actions)

            def keep_finished(new, old):
                return jnp.where(done.reshape((-1,) + (1,) * (new.ndim - 1)), old, new)

            return jax.tree.map(keep_finished, stepped, state)

        self._step = jax.jit(step)
        self._observe = jax.jit(jax.vmap(lambda s: game.observe(s, s.color)))
        self._is_terminal = jax.jit(jax.vmap(game.is_terminal))
        self._terminal_value = jax.jit(jax.vmap(lambda s: game.rewards(s)[s.color]))

    def init(self, batch_size: int) -> GameState:
        """`batch_size` games at the start position."""
        return jax.tree.map(lambda x: jnp.broadcast_to(x, (batch_size,) + x.shape), self._initial)

    def step(self, state: GameState, actions) -> GameState:
        """Apply one action per game. Games that are already over are left unchanged."""
        return self._step(state, jnp.asarray(actions, dtype=jnp.int32))

    def observe(self, state: GameState) -> jax.Array:
        """(batch, 8, 8, 119) float32 observations from each side to move's perspective."""
        return self._observe(state)

    def legal_action_mask(self, state: GameState) -> jax.Array:
        """(batch, 4672) bool."""
        return state.legal_action_mask

    def is_terminal(self, state: GameState) -> jax.Array:
        """(batch,) bool: checkmate, stalemate, insufficient material, threefold repetition, 50-move rule, 512 plies."""
        return self._is_terminal(state)

    def terminal_value(self, state: GameState) -> jax.Array:
        """(batch,) value for the side to move: -1 if checkmated, else 0."""
        return self._terminal_value(state)
