"""A slow, plain-Python AlphaZero search, written independently of mctx, to check ws_zero.search against.

Positions come from the python-chess reference game (Phase 2), not pgx. Same rules as ws_zero.search:
- PUCT: score = Q(s, a) + c(N) * P(s, a) * sqrt(N) / (1 + N(s, a)), c(N) = 1.25 + log((N + 19652 + 1) / 19652);
- Q(s, a) = reward + discount * V(child), discount -1 (the opponent moves next) or 0 (the game is over);
  an untried move has Q = V(node), the position's own current value; V(node) = mean of the values backed up
  through it, its own evaluation included, from the point of view of its side to move;
- N counts a node's own evaluation as a visit, so N(root) = 1 + simulations so far (as in mctx);
- a move that ends the game gives the mover the result; going through a finished position again backs up its
  result unchanged (mctx descends below it into positions worth 0 with discount 0, which comes to the same thing).
"""
import copy
import math

import numpy as np

from ws_zero.game.moves import action_to_move
from ws_zero.game.reference import ReferenceGame

C_INIT, C_BASE = 1.25, 19652


class Node:
    def __init__(self, game, logits=None, terminal=False):
        self.game, self.terminal = game, terminal
        self.visits, self.value_sum = 0, 0.0
        self.children = {}  # action -> Node
        self.rewards = {}  # action -> reward for the mover
        if not terminal:
            legal = game.legal_actions()
            z = np.exp(logits[legal] - logits[legal].max())
            self.priors = dict(zip(legal, z / z.sum()))

    def q(self, action):
        child = self.children.get(action)
        if child is None or child.visits == 0:
            return self.value_sum / self.visits  # untried: as good as this position
        discount = 0.0 if child.terminal else -1.0
        return self.rewards[action] + discount * child.value_sum / child.visits

    def select(self):
        c = C_INIT + math.log((self.visits + C_BASE + 1) / C_BASE)
        def score(a):
            n = self.children[a].visits if a in self.children else 0
            return self.q(a) + c * self.priors[a] * math.sqrt(self.visits) / (1 + n)
        return max(sorted(self.priors), key=score)


def reference_search(fen: str, evaluate_one, num_simulations: int) -> Node:
    """evaluate_one(obs (8, 8, 119)) -> (logits (4672,), value for the side to move)."""
    game = ReferenceGame(fen, repetition_key="pgx")
    logits, value = evaluate_one(game.observation())
    root = Node(game, logits)
    root.visits, root.value_sum = 1, value
    for _ in range(num_simulations):
        node, path = root, []
        while True:
            action = node.select()
            path.append((node, action))
            child = node.children.get(action)
            if child is None:  # expand
                g = copy.deepcopy(node.game)
                g.push(action_to_move(g.board, action))
                over, result = g.outcome()  # result is for the side to move in g
                if over:
                    child = Node(g, terminal=True)
                    node.rewards[action], leaf = -result, 0.0
                else:
                    logits, leaf = evaluate_one(g.observation())
                    child = Node(g, logits)
                    node.rewards[action] = 0.0
                node.children[action] = child
                break
            if child.terminal:  # a finished position again: its result, unchanged
                leaf = 0.0
                break
            node = child
        child.visits += 1
        child.value_sum += leaf
        g_value = leaf
        for parent, action in reversed(path):
            discount = 0.0 if parent.children[action].terminal else -1.0
            g_value = parent.rewards[action] + discount * g_value
            parent.visits += 1
            parent.value_sum += g_value
    return root


def root_visits(root: Node, num_actions: int) -> np.ndarray:
    visits = np.zeros(num_actions)
    for action, child in root.children.items():
        visits[action] = child.visits
    return visits

