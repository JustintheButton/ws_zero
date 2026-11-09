/* ws_zero arena: the chess game as the network sees it.

   A port of ws_zero.game.reference (the python-chess reference that matches pgx exactly) and ws_zero.game.moves:
   - the 119-plane observation, from the side to move's view (row 0 = far rank, column 0 = a-file), NHWC order;
   - the 4,672-action move numbering, action = (file * 8 + rank) * 73 + plane, ranks mirrored for Black;
   - the game's end: checkmate (-1 for the side to move), stalemate, 100 plies without capture or pawn move,
     insufficient material, threefold repetition (pgx's rule: the en passant square counts after every double pawn
     push) and the 512-ply cap (draws, 0).
   Legal moves come from chess.js 0.10.3 (global `Chess`). A Game is immutable: play() returns a new Game that
   points back to the old one, so the search can branch without copying histories. */
(function (factory) {
  const api = factory();
  if (typeof module === "object" && module.exports) module.exports = api;
  else globalThis.WSZ = Object.assign(globalThis.WSZ || {}, api);
})(function () {
  "use strict";
  const MAX_PLIES = 512, HISTORY = 8, PLANES = 119, NUM_ACTIONS = 64 * 73;
  const PIECE_INDEX = { p: 0, n: 1, b: 2, r: 3, q: 4, k: 5 };
  const FILES = "abcdefgh";
  const UNDER_PIECES = ["r", "b", "n"], UNDER_DFILE = [0, 1, -1];

  // (rank change, file change) for planes 9..72, in pgx's order (ws_zero.game.moves._queen_and_knight_deltas).
  const DELTAS = (() => {
    const up = [1, 2, 3, 4, 5, 6, 7], down = [-7, -6, -5, -4, -3, -2, -1], none = [0, 0, 0, 0, 0, 0, 0];
    const rev = (a) => a.slice().reverse();
    const dr = [...down, ...up, ...none, ...none, ...down, ...up, ...rev(up), ...rev(down), -1, 1, -2, 2, -1, 1, -2, 2];
    const df = [...none, ...none, ...down, ...up, ...down, ...up, ...down, ...up, -2, -2, -1, -1, 2, 2, 1, 1];
    return dr.map((r, i) => [r, df[i]]);
  })();
  const PLANE_OF_DELTA = new Map(DELTAS.map(([r, f], i) => [`${r},${f}`, 9 + i]));

  const fileOf = (sq) => FILES.indexOf(sq[0]);
  const rankOf = (sq) => Number(sq[1]) - 1;

  /** The action number of a legal move {from, to, promotion} for the side `turn` ("w" or "b"). */
  function moveToAction(turn, move) {
    const f0 = fileOf(move.from), f1 = fileOf(move.to);
    const rel = (r) => (turn === "w" ? r : 7 - r);
    const r0 = rel(rankOf(move.from)), r1 = rel(rankOf(move.to));
    let plane;
    if (move.promotion && move.promotion !== "q") plane = 3 * UNDER_PIECES.indexOf(move.promotion) + UNDER_DFILE.indexOf(f1 - f0);
    else plane = PLANE_OF_DELTA.get(`${r1 - r0},${f1 - f0}`);
    return (f0 * 8 + r0) * 73 + plane;
  }

  let engine = null;  // one chess.js instance, reloaded from a FEN for every query
  const chess = () => engine || (engine = new globalThis.Chess());

  function parsePieces(fen) {  // [[rank, file, color, type], ...]
    const pieces = [];
    let rank = 7, file = 0;
    for (const ch of fen.split(" ")[0]) {
      if (ch === "/") { rank--; file = 0; continue; }
      if (ch >= "1" && ch <= "8") { file += Number(ch); continue; }
      const lower = ch.toLowerCase();
      pieces.push([rank, file, ch === lower ? "b" : "w", lower]);
      file++;
    }
    return pieces;
  }

  class Game {
    constructor(fen, parent, lastMove) {
      const fields = fen.split(" ");
      this.fen = fen;
      this.parent = parent;
      this.lastMove = lastMove;  // chess.js verbose move that led here (null at the start)
      this.turn = fields[1];
      this.castling = fields[2];
      this.halfmove = Number(fields[4]);
      this.key = fields.slice(0, 4).join(" ");  // repetition identity (pgx: en passant square always shown)
      this.pieces = parsePieces(fen);
      this.plies = parent ? parent.plies + 1 : 0;
      this._legal = null;
      this._outcome = null;
    }

    /** A new game at `fen` (the standard start when omitted), with no history before it. */
    static start(fen) {
      return new Game(fen || "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", null, null);
    }

    /** Legal moves as chess.js verbose moves, each with its `action`, sorted by action. */
    legalMoves() {
      if (!this._legal) {
        const c = chess();
        c.load(this.fen);
        this._legal = c.moves({ verbose: true }).map((m) => Object.assign(m, { action: moveToAction(this.turn, m) }))
          .sort((a, b) => a.action - b.action);
      }
      return this._legal;
    }

    /** The game after `move` ({from, to, promotion} or one of legalMoves()). */
    play(move) {
      const c = chess();
      c.load(this.fen);
      const made = c.move({ from: move.from, to: move.to, promotion: move.promotion || "q" });
      if (!made) throw new Error(`illegal move ${move.from}${move.to} in ${this.fen}`);
      return new Game(c.fen(), this, made);
    }

    /** The positions from the start to this one, oldest first. */
    line() {
      const out = [];
      for (let g = this; g; g = g.parent) out.push(g);
      return out.reverse();
    }

    keyCounts() {
      const counts = new Map();
      for (let g = this; g; g = g.parent) counts.set(g.key, (counts.get(g.key) || 0) + 1);
      return counts;
    }

    /** [game over, value for the side to move]: -1 if checkmated, 0 for a draw or a game in progress. */
    outcome() {
      if (!this._outcome) {
        if (this.legalMoves().length === 0) {
          const c = chess();
          c.load(this.fen);
          this._outcome = [true, c.in_check() ? -1 : 0];
        } else {
          const c = chess();
          c.load(this.fen);
          const drawn = this.halfmove >= 100 || c.insufficient_material() || this.keyCounts().get(this.key) >= 3
            || this.plies >= MAX_PLIES;
          this._outcome = [drawn, 0];
        }
      }
      return this._outcome;
    }

    /** Why the game ended, in words (null while it goes on). */
    endReason() {
      const [over, value] = this.outcome();
      if (!over) return null;
      if (value === -1) return "checkmate";
      if (this.legalMoves().length === 0) return "stalemate";
      if (this.halfmove >= 100) return "fifty-move rule";
      if (this.keyCounts().get(this.key) >= 3) return "threefold repetition";
      if (this.plies >= MAX_PLIES) return "512-ply cap";
      return "insufficient material";
    }

    /** The (8, 8, 119) float32 observation, flattened NHWC: index (row * 8 + column) * 119 + plane. */
    observation() {
      const obs = new Float32Array(64 * PLANES);
      const fill = (plane, v) => { for (let s = 0; s < 64; s++) obs[s * PLANES + plane] = v; };
      const turn = this.turn, counts = this.keyCounts();
      let g = this;
      for (let step = 0; step < HISTORY; step++) {
        const base = step * 14;
        if (!g) { fill(base + 12, 1); continue; }  // before the game started: no pieces, "not repeated"
        for (const [rank, file, color, type] of g.pieces) {
          const r = turn === "w" ? rank : 7 - rank;
          obs[((7 - r) * 8 + file) * PLANES + base + (color === turn ? 0 : 6) + PIECE_INDEX[type]] = 1;
        }
        const repeated = counts.get(g.key) - 1;  // other occurrences of this position in the game so far
        fill(base + 12, repeated === 0 ? 1 : 0);
        fill(base + 13, repeated >= 1 ? 1 : 0);
        g = g.parent;
      }
      const own = turn === "w" ? ["Q", "K"] : ["q", "k"], opp = turn === "w" ? ["q", "k"] : ["Q", "K"];
      fill(112, turn === "b" ? 1 : 0);
      fill(113, this.plies / MAX_PLIES);
      fill(114, this.castling.includes(own[0]) ? 1 : 0);
      fill(115, this.castling.includes(own[1]) ? 1 : 0);
      fill(116, this.castling.includes(opp[0]) ? 1 : 0);
      fill(117, this.castling.includes(opp[1]) ? 1 : 0);
      fill(118, this.halfmove / 100);
      return obs;
    }
  }

  return { Game, moveToAction, NUM_ACTIONS, MAX_PLIES };
});
