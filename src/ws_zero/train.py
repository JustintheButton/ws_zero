"""Optimizer and training step (docs/protocol.md §4). The replay buffer and training loop are Phase 7."""
import jax
import optax

from ws_zero.net import loss_fn


def make_optimizer(learning_rate, momentum: float = 0.9, weight_decay: float = 1e-4) -> optax.GradientTransformation:
    """SGD with heavy-ball momentum and PyTorch-style weight decay (added to the gradient), weights only.

    `learning_rate` may be a number or an optax schedule. Decay of 1e-4 equals the AlphaZero pseudocode's
    1e-4 * l2_loss(weights); biases are not decayed.
    """
    def weights_only(params):
        return {name: not name.endswith("/b") for name in params}

    return optax.chain(optax.add_decayed_weights(weight_decay, mask=weights_only),
                       optax.sgd(learning_rate, momentum=momentum))


def make_train_step(optimizer: optax.GradientTransformation):
    """A jitted step: (params, opt_state, (obs, mask, pi, z)) -> (params, opt_state, loss, loss parts)."""
    @jax.jit
    def step(params, opt_state, batch):
        (loss, parts), grads = jax.value_and_grad(loss_fn, has_aux=True)(params, *batch)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        return optax.apply_updates(params, updates), opt_state, loss, parts

    return step
