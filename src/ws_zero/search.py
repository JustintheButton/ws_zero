"""AlphaZero's search (docs/protocol.md §3) on mctx: the connection between mctx, the chess environment and a network.

mctx needs two things for each batch of games:
- a root (`make_root`): the network's move logits and value at the current positions;
- a recurrent function (`make_recurrent_fn`): given positions and one move per game, the next positions, the
  reward for the player who moved, a discount, and the network's move logits and value at the next positions.

Values are always for the side to move. mctx scores a move as reward + discount x (value of the next position);
in chess the next position belongs to the opponent, so the discount is -1 (the sign flip: good for them is bad for
the mover). A move that ends the game gives the mover the result (+1 for checkmate, 0 for a draw) and discount 0,
so nothing below a finished position counts. Illegal moves get logit ILLEGAL_LOGIT (probability exactly 0).

`evaluate(params, obs) -> (logits (batch, 4672), value (batch,))` is the network: `ws_zero.net.apply`, or a dummy
network such as the Phase 4 gate's material counter.

Move selection inside the tree is AlphaZero's PUCT (mctx's `muzero_policy`, constants 1.25 and 19652) with
AlphaZero's value term (`alphazero_qtransform`) in place of mctx's default. One small difference from the
pseudocode remains: mctx counts the root's own evaluation as a visit, so N(root) is one more than in the
pseudocode (it only changes sqrt(N) in the exploration bonus at the root).
"""
import jax.numpy as jnp
import mctx

from ws_zero.net import ILLEGAL_LOGIT

PB_C_INIT = 1.25  # protocol §3
PB_C_BASE = 19652
DIRICHLET_ALPHA = 0.3
DIRICHLET_FRACTION = 0.25  # self-play only


def _masked(logits, state):
    return jnp.where(state.legal_action_mask, logits, ILLEGAL_LOGIT)


def make_root(env, evaluate, params, state) -> mctx.RootFnOutput:
    """The search root for a batch of positions."""
    logits, value = evaluate(params, env.observe(state))
    return mctx.RootFnOutput(prior_logits=_masked(logits, state), value=value, embedding=state)


def make_recurrent_fn(env, evaluate):
    """mctx's recurrent function for chess: (params, rng, action, state) -> (RecurrentFnOutput, next state)."""
    def recurrent_fn(params, rng, action, state):
        was_over = env.is_terminal(state)  # the search went below a finished position: no further result
        child = env.step(state, action)  # finished games are left unchanged
        over = env.is_terminal(child)
        ends_now = over & ~was_over
        logits, value = evaluate(params, env.observe(child))
        # terminal_value is for the side to move in the finished position (-1 if checkmated); the mover gets minus it.
        reward = jnp.where(ends_now, -env.terminal_value(child), 0.0).astype(jnp.float32)
        out = mctx.RecurrentFnOutput(
            reward=reward,
            discount=jnp.where(over, 0.0, -1.0).astype(jnp.float32),
            prior_logits=_masked(logits, child),
            value=jnp.where(over, 0.0, value).astype(jnp.float32))
        return out, child

    return recurrent_fn


def alphazero_qtransform(tree: mctx.Tree, node_index) -> jnp.ndarray:
    """AlphaZero's value term for choosing a move at a node: the mean value Q(s, a) on the -1..+1 scale.

    Untried moves score 0 ("equal"), the AlphaGo Zero paper's rule (decision 2026-11-08; protocol §3). Illegal moves
    score -inf: mctx excludes them only at the root, so below it this term is what keeps them from being chosen.
    """
    q = tree.qvalues(node_index)
    visits = tree.children_visits[node_index]
    legal = tree.children_prior_logits[node_index] > ILLEGAL_LOGIT / 2
    return jnp.where(legal, jnp.where(visits > 0, q, 0.0), -jnp.inf)


def run_search(env, evaluate, params, rng, state, num_simulations: int,
               dirichlet_fraction: float = DIRICHLET_FRACTION) -> mctx.PolicyOutput:
    """AlphaZero's search from a batch of positions (a new tree each call). Returns mctx's output: the search tree
    and the root visit counts. Choosing the move to play from the visit counts is separate (protocol §3)."""
    root = make_root(env, evaluate, params, state)
    return mctx.muzero_policy(
        params, rng, root, make_recurrent_fn(env, evaluate), num_simulations,
        invalid_actions=~state.legal_action_mask, qtransform=alphazero_qtransform,
        dirichlet_fraction=dirichlet_fraction, dirichlet_alpha=DIRICHLET_ALPHA,
        pb_c_init=PB_C_INIT, pb_c_base=PB_C_BASE)
