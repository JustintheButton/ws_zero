import os
import random
import subprocess
import sys

import numpy as np
import torch

from ws_zero.seeding import derive_seed, seed_everything


def _draw():
    return (random.random(), float(np.random.rand()), float(torch.rand(1).item()))


def test_same_seed_reproduces_python_numpy_and_torch_draws():
    seed_everything(123)
    first = _draw()
    seed_everything(123)
    assert _draw() == first


def test_different_seeds_give_different_draws():
    seed_everything(1)
    a = _draw()
    seed_everything(2)
    b = _draw()
    assert all(x != y for x, y in zip(a, b))


def test_derive_seed_is_deterministic():
    assert derive_seed(7, "selfplay", 3) == derive_seed(7, "selfplay", 3)


def test_derive_seed_gives_distinct_seeds_per_worker_and_base():
    seeds = {derive_seed(7, "selfplay", w) for w in range(1000)}
    assert len(seeds) == 1000
    assert derive_seed(7, "selfplay", 0) != derive_seed(8, "selfplay", 0)
    assert derive_seed(7, "selfplay", 0) != derive_seed(7, "eval", 0)


def test_derive_seed_is_valid_for_numpy_and_torch():
    s = derive_seed(7, "x")
    assert 0 <= s < 2**32
    np.random.seed(s)
    torch.manual_seed(s)


def test_derive_seed_is_stable_across_processes():
    # Must not depend on Python's per-process randomized hash().
    code = "from ws_zero.seeding import derive_seed; print(derive_seed(7, 'selfplay', 3))"
    outs = set()
    for hashseed in ("1", "2"):
        env = {**os.environ, "PYTHONHASHSEED": hashseed}
        out = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, check=True)
        outs.add(out.stdout.strip())
    assert outs == {str(derive_seed(7, "selfplay", 3))}
