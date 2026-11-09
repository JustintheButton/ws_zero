"""AlphaZero's search (docs/protocol.md §3) on mctx: the connection between mctx, the chess environment and a network.

mctx needs two things for each batch of games:
- a root (`make_root`): the network's move logits and value at the current positions;
- a recurrent function (`make_recurrent_fn`): given positions and one move per game, the next positions, the
  reward for the player who moved, a discount, and the network's move logits and value at the next positions.

Values are always for the side to move. mctx scores a move as reward + discount x (value of the next position);
in chess the next position belongs to the opponent, so the discount is -1 (the sign flip: good for them is bad for
the mover). A move that ends the game gives the mover the result (+1 for checkmate, 0 for a draw) and discount 0,
so nothing below a finished position counts. Illegal moves get logit ILLEGAL_LOGIT (probability exactly 0); below
a finished position every move is allowed instead, so that return visits spread out rather than dig a chain.

`evaluate(params, obs) -> (logits (batch, 4672), value (batch,))` is the network: `ws_zero.net.apply`, or a dummy
network such as the Phase 4 gate's material counter.

Move selection inside the tree is AlphaZero's PUCT (mctx's `muzero_policy`, constants 1.25 and 19652) with
AlphaZero's value term, untried moves scored at the position's own value (`alphazero_qtransform`), in place of
mctx's default. One small difference from the
pseudocode remains: mctx counts the root's own evaluation as a visit, so N(root) is one more than in the
pseudocode (it only changes sqrt(N) in the exploration bonus at the root).
"""
import jax
import jax.numpy as jnp
import mctx

from ws_zero.net import ILLEGAL_LOGIT

PB_C_INIT = 1.25  # protocol §3
PB_C_BASE = 19652
DIRICHLET_ALPHA = 0.3
DIRICHLET_FRACTION = 0.25  # self-play only
SAMPLING_PLIES = 30  # self-play samples its move for the first 30 plies


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
        # Below a finished game every move is allowed, all equally likely. Nothing there counts (discount 0), and with
        # every move masked the search took the same slot on each return visit, growing a chain one move deeper
        # each time (100+ moves below a mate in the Phase 4 gate); spread out, the visits stay one move below.
        out = mctx.RecurrentFnOutput(
            reward=reward,
            discount=jnp.where(over, 0.0, -1.0).astype(jnp.float32),
            prior_logits=jnp.where(over[:, None], 0.0, _masked(logits, child)),
            value=jnp.where(over, 0.0, value).astype(jnp.float32))
        return out, child

    return recurrent_fn


def alphazero_qtransform(tree: mctx.Tree, node_index) -> jnp.ndarray:
    """AlphaZero's value term for choosing a move at a node: the mean value Q(s, a) on the -1..+1 scale.

    An untried move scores the position's own current value V(s), from the same side's point of view (decision
    2026-11-08, revised after the Phase 4 gate; protocol §3). A fixed score (0, "equal") was tried first: when the
    side to move was ahead, every tried move looked better than any untried one, and the search did not try the
    mate. Illegal moves score -inf: mctx excludes them only at the root, so below it this term keeps them out.
    """
    q = tree.qvalues(node_index)
    visits = tree.children_visits[node_index]
    legal = tree.children_prior_logits[node_index] > ILLEGAL_LOGIT / 2
    return jnp.where(legal, jnp.where(visits > 0, q, tree.node_values[node_index]), -jnp.inf)


def add_root_noise(rng, prior_logits, legal, fraction: float):
    """AlphaZero's root noise: P = (1 - fraction) p + fraction eta, eta ~ Dirichlet(DIRICHLET_ALPHA) over the legal
    moves only (independent gamma draws on the legal moves, normalized). Returns logits; illegal moves stay masked.

    mctx's built-in noise is drawn over all 4,672 moves, so in chess almost none of it would reach the ~30 legal
    ones; it is switched off. (The pseudocode leaves its gamma draws unnormalized, a slip: the paper says Dirichlet.)
    """
    gamma = jnp.where(legal, jax.random.gamma(rng, DIRICHLET_ALPHA, prior_logits.shape), 0.0)
    eta = gamma / jnp.maximum(gamma.sum(axis=-1, keepdims=True), 1e-30)
    mixed = (1 - fraction) * jax.nn.softmax(prior_logits) + fraction * eta
    return jnp.where(legal, jnp.log(jnp.maximum(mixed, 1e-30)), ILLEGAL_LOGIT)


def run_search(env, evaluate, params, rng, state, num_simulations: int,
               dirichlet_fraction: float = DIRICHLET_FRACTION) -> mctx.PolicyOutput:
    """AlphaZero's search from a batch of positions (a new tree each call). Returns mctx's output: the search tree
    and the root visit counts. Choosing the move to play from the visit counts is separate (protocol §3)."""
    noise_key, search_key = jax.random.split(rng)
    root = make_root(env, evaluate, params, state)
    if dirichlet_fraction:
        root = root.replace(prior_logits=add_root_noise(noise_key, root.prior_logits, state.legal_action_mask,
                                                        dirichlet_fraction))
    return mctx.muzero_policy(
        params, search_key, root, make_recurrent_fn(env, evaluate), num_simulations,
        invalid_actions=~state.legal_action_mask, qtransform=alphazero_qtransform,
        dirichlet_fraction=0.0, pb_c_init=PB_C_INIT, pb_c_base=PB_C_BASE)


def choose_moves(rng, visit_counts, plies, sample: bool):
    """The move to play from the root visit counts (protocol §3).

    sample=True (self-play): for the first SAMPLING_PLIES plies of the game, a move drawn with probability
    proportional to its visit count; after that, the most-visited move. sample=False (evaluation): the most-visited
    move from the first ply. Ties go to the lowest action number.
    """
    best = jnp.argmax(visit_counts, axis=-1)
    if not sample:
        return best
    drawn = jax.random.categorical(rng, jnp.log(visit_counts), axis=-1)  # log 0 = -inf: unvisited never drawn
    return jnp.where(plies < SAMPLING_PLIES, drawn, best)


def search_moves(env, evaluate, params, rng, state, num_simulations: int, self_play: bool):
    """Search every position in the batch and choose the moves to play: (actions, mctx output).

    Self-play: root Dirichlet noise, and early moves sampled. Evaluation: no noise, the most-visited move.
    A game that is already over gets a meaningless action, which the environment ignores.
    """
    search_key, choice_key = jax.random.split(rng)
    out = run_search(env, evaluate, params, search_key, state, num_simulations,
                     dirichlet_fraction=DIRICHLET_FRACTION if self_play else 0.0)
    actions = choose_moves(choice_key, out.search_tree.summary().visit_counts, state.step_count, sample=self_play)
    return actions, out
