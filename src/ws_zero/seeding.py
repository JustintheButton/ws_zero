"""Seed control: one base seed per run, with stable derived seeds for workers and sub-tasks.

JAX has no global generator: JAX code takes explicit keys, e.g. `jax.random.key(derive_seed(seed, "init"))`.
"""
import hashlib
import random

import numpy as np


def seed_everything(seed: int) -> None:
    """Seed Python's `random` and NumPy's global generator."""
    random.seed(seed)
    np.random.seed(seed)


def derive_seed(base: int, *keys) -> int:
    """A seed in [0, 2**32) determined by `base` and `keys`, identical in every process and on every machine.

    Example: derive_seed(run_seed, "selfplay", worker_id) gives each self-play worker its own stream.
    """
    material = repr((base, *keys)).encode()
    return int.from_bytes(hashlib.sha256(material).digest()[:4], "little")
