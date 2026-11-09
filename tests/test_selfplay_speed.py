"""scripts/selfplay_speed.py: how deep the search trees were, read from mctx's parent table."""
import importlib.util
from pathlib import Path

import numpy as np

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "selfplay_speed.py"


def _load():
    spec = importlib.util.spec_from_file_location("selfplay_speed", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_tree_depths_counts_moves_below_the_root_and_skips_nodes_never_created():
    # Two games, root + 5 nodes each. -1 marks the root and any node the search never created (a look-ahead that
    # returned to a finished game creates none).
    parents = np.array([
        [-1, 0, 1, 2, 0, 4],    # a line 3 moves deep (0-1-2-3) and one 2 deep (0-4-5)
        [-1, 0, 0, -1, 1, -1],  # nodes 3 and 5 never created; the deepest line is 0-1-4
    ])
    deepest, mean = _load().tree_depths(parents)
    np.testing.assert_array_equal(deepest, [3, 2])
    np.testing.assert_allclose(mean, [(1 + 2 + 3 + 1 + 2) / 5, (1 + 1 + 2) / 3])
