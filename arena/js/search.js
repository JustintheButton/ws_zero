/* ws_zero arena: AlphaZero's search, as in ws_zero.search (which agrees with tests/reference_search.py, of which
   this is a port, visit for visit).

   - PUCT: score = Q(s, a) + c(N) * P(s, a) * sqrt(N) / (1 + N(s, a)), c(N) = 1.25 + log((N + 19652 + 1) / 19652);
   - Q(s, a) = reward + discount * V(child): discount -1 (the opponent moves next) or 0 (the game is over); an untried
     move has Q = V(node), the position's own current value; V(node) = mean of the values backed up through it, its
     own evaluation included, from its side to move's point of view;
   - N counts a node's own evaluation, so N(root) = 1 + simulations so far;
   - a move that ends the game gives the mover the result; going through a finished position again backs up its
     result unchanged; no root noise (play and evaluation); ties go to the lowest action number.

   evaluate(game) -> Promise<{logits, value}> or {logits, value}: logits indexed by action (length 4,672), value for
   the side to move in [-1, 1]. */
(function (factory) {
  const api = factory();
  if (typeof module === "object" && module.exports) module.exports = api;
  else globalThis.WSZ = Object.assign(globalThis.WSZ || {}, api);
})(function () {
  "use strict";
  const C_INIT = 1.25, C_BASE = 19652;

  class Node {
    constructor(game, logits, terminal) {
      this.game = game;
      this.terminal = terminal;
      this.visits = 0;
      this.valueSum = 0;
      this.children = new Map();  // action -> Node
      this.rewards = new Map();  // action -> reward for the mover
      if (!terminal) {
        this.moves = game.legalMoves();  // sorted by action
        let max = -Infinity;
        for (const m of this.moves) max = Math.max(max, logits[m.action]);
        const z = this.moves.map((m) => Math.exp(logits[m.action] - max));
        const total = z.reduce((a, b) => a + b, 0);
        this.priors = z.map((x) => x / total);
      }
    }

    q(action) {
      const child = this.children.get(action);
      if (!child || child.visits === 0) return this.valueSum / this.visits;  // untried: as good as this position
      return this.rewards.get(action) + (child.terminal ? 0 : -1) * child.valueSum / child.visits;
    }

    select() {
      const c = C_INIT + Math.log((this.visits + C_BASE + 1) / C_BASE);
      const sqrtN = Math.sqrt(this.visits);
      let best = 0, bestScore = -Infinity;
      for (let i = 0; i < this.moves.length; i++) {
        const a = this.moves[i].action, child = this.children.get(a);
        const s = this.q(a) + c * this.priors[i] * sqrtN / (1 + (child ? child.visits : 0));
        if (s > bestScore) { bestScore = s; best = i; }
      }
      return this.moves[best];
    }
  }

  /** Search `game` (not finished) with `simulations` look-aheads. Returns the root's moves with their visit counts
      and values, sorted by action, the move to play (most visits; ties to the lowest action), and the root value. */
  async function search(game, evaluate, simulations, onProgress) {
    const first = await evaluate(game);
    const root = new Node(game, first.logits, false);
    root.visits = 1;
    root.valueSum = first.value;
    for (let sim = 0; sim < simulations; sim++) {
      let node = root, child, leaf;
      const path = [];
      for (;;) {
        const move = node.select();
        path.push([node, move.action]);
        child = node.children.get(move.action);
        if (!child) {  // expand
          const next = node.game.play(move);
          const [over, result] = next.outcome();  // result is for the side to move in `next`
          if (over) {
            child = new Node(next, null, true);
            node.rewards.set(move.action, -result);
            leaf = 0;
          } else {
            const ev = await evaluate(next);
            child = new Node(next, ev.logits, false);
            node.rewards.set(move.action, 0);
            leaf = ev.value;
          }
          node.children.set(move.action, child);
          break;
        }
        if (child.terminal) { leaf = 0; break; }  // a finished position again: its result, unchanged
        node = child;
      }
      child.visits += 1;
      child.valueSum += leaf;
      let g = leaf;
      for (let i = path.length - 1; i >= 0; i--) {
        const [parent, action] = path[i];
        g = parent.rewards.get(action) + (parent.children.get(action).terminal ? 0 : -1) * g;
        parent.visits += 1;
        parent.valueSum += g;
      }
      if (onProgress && (sim + 1) % 25 === 0) await onProgress(sim + 1, simulations);
    }
    const moves = root.moves.map((m) => {
      const ch = root.children.get(m.action);
      return { move: m, visits: ch ? ch.visits : 0, value: ch && ch.visits ? root.q(m.action) : null };
    });
    let best = moves[0];
    for (const m of moves) if (m.visits > best.visits) best = m;
    return { moves, best: best.move, rootValue: root.valueSum / root.visits, prior: first };
  }

  /** "Instinct only": the network's own top move, no search (protocol §8.3's secondary agent). */
  async function instinct(game, evaluate) {
    const ev = await evaluate(game);
    let best = null;
    for (const m of game.legalMoves()) if (!best || ev.logits[m.action] > ev.logits[best.action]) best = m;
    return { best, rootValue: ev.value, prior: ev };
  }

  /** The Phase 4 gate's stand-in: every move equally likely, value tanh((own material - opponent's) / 4). */
  const PIECE_VALUE = { p: 1, n: 3, b: 3, r: 5, q: 9, k: 0 };
  function materialCounter(game) {
    let diff = 0;
    for (const [, , color, type] of game.pieces) diff += (color === game.turn ? 1 : -1) * PIECE_VALUE[type];
    return { logits: new Float64Array(64 * 73), value: Math.tanh(diff / 4) };
  }

  return { search, instinct, materialCounter };
});
