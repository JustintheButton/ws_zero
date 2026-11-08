"""The AlphaZero network at the protocol's scale (docs/protocol.md §2), in plain JAX, with batch norm.

The weights are a flat dict of named arrays. Every conv except the policy output is followed by batch norm, as in
AlphaZero; the batch norm offset plays the role of the conv's bias, so those convs have none:

    stem/w (3, 3, 119, 64), stem/bn/*                         3x3 conv, BN, ReLU
    block{i}/conv1/w (3, 3, 64, 64), block{i}/bn1/*            residual block i = 0..5:
    block{i}/conv2/w (3, 3, 64, 64), block{i}/bn2/*            x <- ReLU(x + BN2(conv2(ReLU(BN1(conv1(x))))))
    policy/conv/w (3, 3, 64, 64), policy/bn/*                 3x3 conv, BN, ReLU
    policy/out/w (1, 1, 64, 73), policy/out/b (73)            1x1 conv -> 8 x 8 x 73 logits
    value/conv/w (1, 1, 64, 1), value/bn/*                    1x1 conv, BN, ReLU, flattened row by row
    value/fc1/w (64, 256), value/fc1/b (256)                  linear, ReLU
    value/fc2/w (256, 1), value/fc2/b (1)                     linear, tanh

Each batch norm `name` has a trainable `name/scale` and `name/offset` and running statistics `name/mean` and
`name/var` (STAT_NAMES), which are not trained by gradient: in training, batch norm uses the batch's statistics and
moves the running ones towards them (BN_MOMENTUM); in play and evaluation (`apply`) it uses the running ones.

Layouts: observations are (batch, 8, 8, 119) as in docs/encoding.md (row 0 = far rank, column 0 = a-file). Conv
weights are (kernel row, kernel column, in, out) with "same" zero padding; kernel tap [1, 1] is the centre. Linear
weights are (in, out). The value head's flattened input has square (row, column) at index row * 8 + column. The
policy logit for observation square (row, column) and move type p is action (column * 8 + 7 - row) * 73 + p.

History (protocol §10): protocol v1.0 removed batch norm in every arm. Without it the value head's single ReLU
channel died, then (with that ReLU removed) the value output saturated after rare explosive training steps.
Batch norm was restored on 2026-11-08 (Justin, before any results), with AlphaZero's original value head.
"""
import jax
import jax.numpy as jnp
import numpy as np

BLOCKS = 6
FILTERS = 64
INPUT_PLANES = 119
MOVE_TYPES = 73
VALUE_HIDDEN = 256
ILLEGAL_LOGIT = -1e9  # masked logits; exp() of it is exactly 0 in float32
BN_EPS = 1e-5  # PyTorch's default
BN_MOMENTUM = 0.1  # weight of the new batch in the running statistics (PyTorch's default)

BN_LAYERS = ["stem/bn"] + [f"block{i}/bn{j}" for i in range(BLOCKS) for j in (1, 2)] + ["policy/bn", "value/bn"]


def _param_shapes() -> dict[str, tuple[int, ...]]:
    f = FILTERS
    shapes = {"stem/w": (3, 3, INPUT_PLANES, f)}
    for i in range(BLOCKS):
        shapes[f"block{i}/conv1/w"] = (3, 3, f, f)
        shapes[f"block{i}/conv2/w"] = (3, 3, f, f)
    shapes.update({
        "policy/conv/w": (3, 3, f, f),
        "policy/out/w": (1, 1, f, MOVE_TYPES), "policy/out/b": (MOVE_TYPES,),
        "value/conv/w": (1, 1, f, 1),
        "value/fc1/w": (64, VALUE_HIDDEN), "value/fc1/b": (VALUE_HIDDEN,),
        "value/fc2/w": (VALUE_HIDDEN, 1), "value/fc2/b": (1,),
    })
    for name in BN_LAYERS:
        channels = 1 if name == "value/bn" else f
        for field in ("scale", "offset", "mean", "var"):
            shapes[f"{name}/{field}"] = (channels,)
    return shapes


PARAM_SHAPES = _param_shapes()
STAT_NAMES = frozenset(n for n in PARAM_SHAPES if n.endswith(("/mean", "/var")))

# Zero-initialized in the random arm (protocol §4): the last layer of every residual branch (its batch norm scale,
# so the block starts as the identity) and of each head.
ZERO_INIT = frozenset([f"block{i}/bn2/scale" for i in range(BLOCKS)] + ["policy/out/w", "value/fc2/w"])


def num_params(shapes: dict[str, tuple[int, ...]]) -> int:
    return sum(int(np.prod(s)) for s in shapes.values())


def blank_params() -> dict[str, jax.Array]:
    """Every weight and bias 0 and every batch norm the identity (scale 1, offset 0, running mean 0, variance 1)."""
    return {name: jnp.ones(shape) if name.endswith(("/scale", "/var")) else jnp.zeros(shape)
            for name, shape in PARAM_SHAPES.items()}


def init_params(key: jax.Array) -> dict[str, jax.Array]:
    """Random-arm initialization: He (Kaiming) normal weights, zero biases, identity batch norms, ZERO_INIT zero."""
    params = blank_params()
    weights = [n for n in PARAM_SHAPES if n.endswith("/w")]
    for k, name in zip(jax.random.split(key, len(weights)), weights):
        shape = PARAM_SHAPES[name]
        fan_in = int(np.prod(shape[:-1]))
        params[name] = jax.random.normal(k, shape, jnp.float32) * np.sqrt(2.0 / fan_in)
    for name in ZERO_INIT:
        params[name] = jnp.zeros(PARAM_SHAPES[name], jnp.float32)
    return params


def batch_norm(x, scale, offset, mean, var, train: bool):
    """Per-channel batch norm over (batch, row, column). Returns (output, running mean, running variance).

    train=True normalizes with the batch's statistics and returns running statistics moved towards them;
    train=False normalizes with the given running statistics and returns them unchanged.
    """
    mean, var = jax.lax.stop_gradient(mean), jax.lax.stop_gradient(var)
    if train:
        batch_mean, batch_var = jnp.mean(x, axis=(0, 1, 2)), jnp.var(x, axis=(0, 1, 2))
        y = (x - batch_mean) * jax.lax.rsqrt(batch_var + BN_EPS) * scale + offset
        new_mean = (1 - BN_MOMENTUM) * mean + BN_MOMENTUM * jax.lax.stop_gradient(batch_mean)
        new_var = (1 - BN_MOMENTUM) * var + BN_MOMENTUM * jax.lax.stop_gradient(batch_var)
        return y, new_mean, new_var
    return (x - mean) * jax.lax.rsqrt(var + BN_EPS) * scale + offset, mean, var


def _conv(x: jax.Array, w: jax.Array) -> jax.Array:
    return jax.lax.conv_general_dilated(x, w, window_strides=(1, 1), padding="SAME",
                                        dimension_numbers=("NHWC", "HWIO", "NHWC"))


def policy_to_actions(x: jax.Array) -> jax.Array:
    """(batch, 8, 8, 73) per-square logits -> (batch, 4672) in action order (docs/encoding.md)."""
    x = x[:, ::-1]  # row r -> rank counted from the mover's side, 7 - r
    x = jnp.transpose(x, (0, 2, 1, 3))  # (batch, column, rank, plane)
    return x.reshape(x.shape[0], -1)


def _forward(params, obs, train: bool):
    """Returns logits, value, new running statistics, and the value head's internals (for value_head_health)."""
    stats = {}

    def conv_bn(x, conv, bn):
        y, stats[f"{bn}/mean"], stats[f"{bn}/var"] = batch_norm(
            _conv(x, params[f"{conv}/w"]), params[f"{bn}/scale"], params[f"{bn}/offset"],
            params[f"{bn}/mean"], params[f"{bn}/var"], train)
        return y

    x = jax.nn.relu(conv_bn(obs, "stem", "stem/bn"))
    for i in range(BLOCKS):
        h = jax.nn.relu(conv_bn(x, f"block{i}/conv1", f"block{i}/bn1"))
        x = jax.nn.relu(x + conv_bn(h, f"block{i}/conv2", f"block{i}/bn2"))
    p = jax.nn.relu(conv_bn(x, "policy/conv", "policy/bn"))
    logits = policy_to_actions(_conv(p, params["policy/out/w"]) + params["policy/out/b"])
    value_map = jax.nn.relu(conv_bn(x, "value/conv", "value/bn")).reshape(x.shape[0], 64)
    hidden = value_map @ params["value/fc1/w"] + params["value/fc1/b"]
    v = jnp.tanh(jax.nn.relu(hidden) @ params["value/fc2/w"] + params["value/fc2/b"])[:, 0]
    return logits, v, stats, (value_map, hidden)


def apply(params: dict[str, jax.Array], obs: jax.Array) -> tuple[jax.Array, jax.Array]:
    """Play and evaluation: obs (batch, 8, 8, 119) -> policy logits (batch, 4672) and value (batch,) in [-1, 1] for
    the side to move. Batch norm uses the running statistics, so each position's output is independent of the batch."""
    logits, v, _, _ = _forward(params, obs, train=False)
    return logits, v


def apply_train(params, obs):
    """Training: batch norm uses the batch's statistics. Returns logits, value and the new running statistics."""
    logits, v, stats, _ = _forward(params, obs, train=True)
    return logits, v, stats


def value_head_health(params: dict[str, jax.Array], obs: jax.Array) -> dict[str, jax.Array]:
    """Signs of a broken value head, logged during training (protocol §10), in play mode.

    conv_active: fraction of (position, square) pairs where the value map's ReLU is on (the original failure: 0).
    fc1_active: fraction of (position, unit) pairs where the 256-unit layer is on; 0 means no gradient reaches it.
    saturated: fraction of positions with |value| > 0.99, where tanh passes almost no gradient.
    value_std: spread of the value across positions; 0 means a constant value (expected only at initialization).
    """
    _, v, _, (value_map, hidden) = _forward(params, obs, train=False)
    return {"conv_active": jnp.mean(value_map > 0), "fc1_active": jnp.mean(hidden > 0),
            "saturated": jnp.mean(jnp.abs(v) > 0.99), "value_std": jnp.std(v)}


def masked_log_softmax(logits: jax.Array, mask: jax.Array) -> jax.Array:
    """Log-probabilities over legal moves only; illegal moves get probability exactly 0."""
    return jax.nn.log_softmax(jnp.where(mask, logits, ILLEGAL_LOGIT), axis=-1)


def loss_fn(params, obs, mask, pi, z):
    """AlphaZero loss without the L2 term (weight decay is in the optimizer): mean (z - v)^2 - pi . log p.

    Computed in training mode. Returns (loss, {"value", "policy", "stats"}), "stats" being the new running
    statistics, which the training step stores.
    """
    logits, v, stats = apply_train(params, obs)
    value_loss = jnp.mean((z - v) ** 2)
    policy_loss = -jnp.mean(jnp.sum(pi * masked_log_softmax(logits, mask), axis=-1))
    return value_loss + policy_loss, {"value": value_loss, "policy": policy_loss, "stats": stats}
