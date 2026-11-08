"""Optimizer and training step (docs/protocol.md §4). The replay buffer and training loop are Phase 7."""
import jax
import optax

from ws_zero.net import loss_fn


def make_optimizer(learning_rate, momentum: float = 0.9, weight_decay: float = 1e-4) -> optax.GradientTransformation:
    """SGD with heavy-ball momentum and PyTorch-style weight decay (added to the gradient), weights only.

    `learning_rate` may be a number or an optax schedule. Decay of 1e-4 equals the AlphaZero pseudocode's
    1e-4 * l2_loss(weights). Only conv and linear weights ("/w") are decayed: not biases, not batch norm scales and
    offsets (common practice), and not running statistics (which get no gradient and are set by the training step).
    """
    def weights_only(params):
        return {name: name.endswith("/w") for name in params}

    return optax.chain(optax.add_decayed_weights(weight_decay, mask=weights_only),
                       optax.sgd(learning_rate, momentum=momentum))


def make_train_step(optimizer: optax.GradientTransformation):
    """A jitted step: (params, opt_state, (obs, mask, pi, z)) -> (params, opt_state, loss, parts).

    parts: value loss, policy loss, and the gradient's global norm (before any clipping). The batch norm running
    statistics computed in the forward pass replace the old ones.
    """
    @jax.jit
    def step(params, opt_state, batch):
        (loss, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(params, *batch)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = {**optax.apply_updates(params, updates), **aux["stats"]}
        parts = {"value": aux["value"], "policy": aux["policy"], "grad_norm": optax.tree.norm(grads)}
        return params, opt_state, loss, parts

    return step
