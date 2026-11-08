"""Seed control: one base seed per run, with stable derived seeds for workers and sub-tasks."""
import hashlib
import random

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    """Seed Python's `random`, NumPy's global generator, and PyTorch (CPU and all GPUs)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # also seeds every CUDA device


def derive_seed(base: int, *keys) -> int:
    """A seed in [0, 2**32) determined by `base` and `keys`, identical in every process and on every machine.

    Example: derive_seed(run_seed, "selfplay", worker_id) gives each self-play worker its own stream.
    """
    material = repr((base, *keys)).encode()
    return int.from_bytes(hashlib.sha256(material).digest()[:4], "little")
