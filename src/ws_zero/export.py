"""Export a network for the browser (the arena page): batch norm folded into the convs, float32 arrays in one file.

In play (`ws_zero.net.apply`) batch norm is a fixed per-channel affine map, so conv -> batch norm becomes one conv
with a bias: w' = w * s and b' = offset - mean * s, where s = scale / sqrt(var + BN_EPS). The folded network has
plain layers only:
    stem/{w,b}; block{i}/conv1/{w,b}, block{i}/conv2/{w,b}; policy/conv/{w,b}; policy/out/{w,b};
    value/conv/{w,b}; value/fc1/{w,b}; value/fc2/{w,b}
with the same layouts as ws_zero.net (conv weights (kernel row, kernel column, in, out), observations NHWC).

A model is two files: `<path>.bin` (the arrays, float32 little-endian, back to back) and `<path>.json` (caller's
fields, plus "arrays": [[name, shape, offset in floats], ...]).
"""
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from ws_zero.net import BLOCKS, BN_EPS, _conv, policy_to_actions

_CONV_BN = [("stem", "stem/bn")] + [(f"block{i}/conv{j}", f"block{i}/bn{j}") for i in range(BLOCKS) for j in (1, 2)] \
    + [("policy/conv", "policy/bn"), ("value/conv", "value/bn")]


def fold_batch_norm(params: dict) -> dict[str, np.ndarray]:
    """The play-mode network with every batch norm folded into the conv before it."""
    folded = {}
    for conv, bn in _CONV_BN:
        s = np.asarray(params[f"{bn}/scale"]) / np.sqrt(np.asarray(params[f"{bn}/var"]) + BN_EPS)
        folded[f"{conv}/w"] = np.asarray(params[f"{conv}/w"]) * s
        folded[f"{conv}/b"] = np.asarray(params[f"{bn}/offset"]) - np.asarray(params[f"{bn}/mean"]) * s
    for name in ("policy/out", "value/fc1", "value/fc2"):
        folded[f"{name}/w"] = np.asarray(params[f"{name}/w"])
        folded[f"{name}/b"] = np.asarray(params[f"{name}/b"])
    return {k: v.astype(np.float32) for k, v in folded.items()}


@jax.jit
def apply_folded(folded: dict, obs: jax.Array) -> tuple[jax.Array, jax.Array]:
    """The folded network's forward pass (the reference for the browser's): logits (batch, 4672), value (batch,)."""
    def conv(x, name):
        return _conv(x, folded[f"{name}/w"]) + folded[f"{name}/b"]

    x = jax.nn.relu(conv(obs, "stem"))
    for i in range(BLOCKS):
        h = jax.nn.relu(conv(x, f"block{i}/conv1"))
        x = jax.nn.relu(x + conv(h, f"block{i}/conv2"))
    logits = policy_to_actions(conv(jax.nn.relu(conv(x, "policy/conv")), "policy/out"))
    value_map = jax.nn.relu(conv(x, "value/conv")).reshape(x.shape[0], 64)
    hidden = jax.nn.relu(value_map @ folded["value/fc1/w"] + folded["value/fc1/b"])
    return logits, jnp.tanh(hidden @ folded["value/fc2/w"] + folded["value/fc2/b"])[:, 0]


def model_file(path: Path, suffix: str) -> Path:
    """`<path><suffix>`, added to the whole name: a model id may contain dots ("random-seed0-1.5k")."""
    path = Path(path)
    return path.with_name(path.name + suffix)


def write_model(path: Path, params: dict, fields: dict) -> None:
    """Write `<path>.bin` and `<path>.json` for the folded network of `params`."""
    path = Path(path)
    arrays, offset, chunks = [], 0, []
    for name, array in fold_batch_norm(params).items():
        arrays.append([name, list(array.shape), offset])
        chunks.append(array.astype("<f4").tobytes())
        offset += array.size
    model_file(path, ".bin").write_bytes(b"".join(chunks))
    model_file(path, ".json").write_text(json.dumps({**fields, "arrays": arrays}))


def read_model(path: Path) -> dict[str, np.ndarray]:
    """Read back the folded arrays written by `write_model`."""
    path = Path(path)
    manifest = json.loads(model_file(path, ".json").read_text())
    data = np.frombuffer(model_file(path, ".bin").read_bytes(), dtype="<f4")
    return {name: data[offset:offset + int(np.prod(shape))].reshape(shape) for name, shape, offset in manifest["arrays"]}
