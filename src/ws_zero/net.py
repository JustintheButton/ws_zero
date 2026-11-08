"""The AlphaZero network at the protocol's scale (docs/protocol.md §2), in plain JAX.

The weights are a flat dict of named arrays, the same names the WSP prompt uses:

    stem/w (3, 3, 119, 64), stem/b (64)                     3x3 conv, ReLU
    block{i}/conv1/w (3, 3, 64, 64), block{i}/conv1/b (64)   residual block i = 0..5:
    block{i}/conv2/w (3, 3, 64, 64), block{i}/conv2/b (64)   x <- ReLU(x + conv2(ReLU(conv1(x))))
    policy/conv/w (3, 3, 64, 64), policy/conv/b (64)        3x3 conv, ReLU
    policy/out/w (1, 1, 64, 73), policy/out/b (73)          1x1 conv -> 8 x 8 x 73 logits
    value/conv/w (1, 1, 64, 1), value/conv/b (1)            1x1 conv (no ReLU), flattened row by row
    value/fc1/w (64, 256), value/fc1/b (256)                linear, ReLU
    value/fc2/w (256, 1), value/fc2/b (1)                   linear, tanh

Layouts: observations are (batch, 8, 8, 119) as in docs/encoding.md (row 0 = far rank, column 0 = a-file). Conv
weights are (kernel row, kernel column, in, out) with "same" zero padding; kernel tap [1, 1] is the centre. Linear
weights are (in, out). The value head's flattened input has square (row, column) at index row * 8 + column. The
policy logit for observation square (row, column) and move type p is action (column * 8 + 7 - row) * 73 + p.

Amendment (Justin, 2026-11-08, before any results): AlphaZero's value head is conv -> batch norm -> ReLU. With batch
norm removed, that single ReLU channel died in every seed under realistic training (Phase 3 report), so the value
conv is linear here; everything else follows the protocol.
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


def _param_shapes() -> dict[str, tuple[int, ...]]:
    f = FILTERS
    shapes = {"stem/w": (3, 3, INPUT_PLANES, f), "stem/b": (f,)}
    for i in range(BLOCKS):
        for conv in ("conv1", "conv2"):
            shapes[f"block{i}/{conv}/w"] = (3, 3, f, f)
            shapes[f"block{i}/{conv}/b"] = (f,)
    shapes.update({
        "policy/conv/w": (3, 3, f, f), "policy/conv/b": (f,),
        "policy/out/w": (1, 1, f, MOVE_TYPES), "policy/out/b": (MOVE_TYPES,),
        "value/conv/w": (1, 1, f, 1), "value/conv/b": (1,),
        "value/fc1/w": (64, VALUE_HIDDEN), "value/fc1/b": (VALUE_HIDDEN,),
        "value/fc2/w": (VALUE_HIDDEN, 1), "value/fc2/b": (1,),
    })
    return shapes


PARAM_SHAPES = _param_shapes()

# Zero-initialized weights in the random arm (protocol §4): the last layer of every residual branch and of each head.
ZERO_INIT = frozenset([f"block{i}/conv2/w" for i in range(BLOCKS)] + ["policy/out/w", "value/fc2/w"])


def num_params(shapes: dict[str, tuple[int, ...]]) -> int:
    return sum(int(np.prod(s)) for s in shapes.values())


def init_params(key: jax.Array) -> dict[str, jax.Array]:
    """Random-arm initialization: He (Kaiming) normal weights, zero biases, ZERO_INIT weights zero."""
    params = {}
    for k, (name, shape) in zip(jax.random.split(key, len(PARAM_SHAPES)), PARAM_SHAPES.items()):
        if name.endswith("/b") or name in ZERO_INIT:
            params[name] = jnp.zeros(shape, jnp.float32)
        else:
            fan_in = int(np.prod(shape[:-1]))
            params[name] = jax.random.normal(k, shape, jnp.float32) * np.sqrt(2.0 / fan_in)
    return params


def _conv(x: jax.Array, w: jax.Array, b: jax.Array) -> jax.Array:
    y = jax.lax.conv_general_dilated(x, w, window_strides=(1, 1), padding="SAME",
                                     dimension_numbers=("NHWC", "HWIO", "NHWC"))
    return y + b


def policy_to_actions(x: jax.Array) -> jax.Array:
    """(batch, 8, 8, 73) per-square logits -> (batch, 4672) in action order (docs/encoding.md)."""
    x = x[:, ::-1]  # row r -> rank counted from the mover's side, 7 - r
    x = jnp.transpose(x, (0, 2, 1, 3))  # (batch, column, rank, plane)
    return x.reshape(x.shape[0], -1)


def trunk(params: dict[str, jax.Array], obs: jax.Array) -> jax.Array:
    """Stem and residual blocks: (batch, 8, 8, 119) -> (batch, 8, 8, 64)."""
    x = jax.nn.relu(_conv(obs, params["stem/w"], params["stem/b"]))
    for i in range(BLOCKS):
        h = jax.nn.relu(_conv(x, params[f"block{i}/conv1/w"], params[f"block{i}/conv1/b"]))
        x = jax.nn.relu(x + _conv(h, params[f"block{i}/conv2/w"], params[f"block{i}/conv2/b"]))
    return x


def apply(params: dict[str, jax.Array], obs: jax.Array) -> tuple[jax.Array, jax.Array]:
    """obs (batch, 8, 8, 119) -> policy logits (batch, 4672) and value (batch,) in [-1, 1] for the side to move."""
    x = trunk(params, obs)
    p = jax.nn.relu(_conv(x, params["policy/conv/w"], params["policy/conv/b"]))
    logits = policy_to_actions(_conv(p, params["policy/out/w"], params["policy/out/b"]))
    v = _value_hidden(params, x)
    v = jnp.tanh(v @ params["value/fc2/w"] + params["value/fc2/b"])
    return logits, v[:, 0]


def _value_hidden_preactivation(params, x):
    v = _conv(x, params["value/conv/w"], params["value/conv/b"]).reshape(x.shape[0], 64)
    return v @ params["value/fc1/w"] + params["value/fc1/b"]


def _value_hidden(params, x):
    return jax.nn.relu(_value_hidden_preactivation(params, x))


def value_head_health(params: dict[str, jax.Array], obs: jax.Array) -> dict[str, jax.Array]:
    """Signs of a broken value head, logged during training (protocol §10).

    fc1_active: fraction of (position, unit) pairs where the 256-unit layer is on; 0 means no gradient reaches it.
    saturated: fraction of positions with |value| > 0.99, where tanh passes almost no gradient.
    value_std: spread of the value across positions; 0 means a constant value (expected only at initialization).
    """
    x = trunk(params, obs)
    hidden = _value_hidden_preactivation(params, x)
    v = jnp.tanh(jax.nn.relu(hidden) @ params["value/fc2/w"] + params["value/fc2/b"])[:, 0]
    return {"fc1_active": jnp.mean(hidden > 0), "saturated": jnp.mean(jnp.abs(v) > 0.99), "value_std": jnp.std(v)}


def masked_log_softmax(logits: jax.Array, mask: jax.Array) -> jax.Array:
    """Log-probabilities over legal moves only; illegal moves get probability exactly 0."""
    return jax.nn.log_softmax(jnp.where(mask, logits, ILLEGAL_LOGIT), axis=-1)


def loss_fn(params, obs, mask, pi, z):
    """AlphaZero loss without the L2 term (weight decay is in the optimizer): mean (z - v)^2 - pi . log p."""
    logits, v = apply(params, obs)
    value_loss = jnp.mean((z - v) ** 2)
    policy_loss = -jnp.mean(jnp.sum(pi * masked_log_softmax(logits, mask), axis=-1))
    return value_loss + policy_loss, {"value": value_loss, "policy": policy_loss}
