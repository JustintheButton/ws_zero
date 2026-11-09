/* Check the arena's browser code against cases computed in Python (tests/test_arena_js.py writes them).

     node arena/tests/check.js <cases.json>

   cases.json: {encoding: [...], networks: [{model: <path without suffix>, cases: [...]}], searches: [...]}.
   Needs build/vendor/chess.min.js and build/vendor/tf.min.js (chess.js 0.10.3, TensorFlow.js 4.22.0). Prints one
   line per check and exits 1 if any fails. */
"use strict";
const fs = require("fs");
const path = require("path");

const ROOT = path.resolve(__dirname, "..", "..");
globalThis.Chess = require(path.join(ROOT, "build/vendor/chess.min.js")).Chess;
const { Game } = require(path.join(ROOT, "arena/js/chessgame.js"));
const { search, materialCounter } = require(path.join(ROOT, "arena/js/search.js"));

const cases = JSON.parse(fs.readFileSync(process.argv[2], "utf8"));
let failures = 0;
const fail = (msg) => { failures++; console.log("FAIL " + msg); };
const lineOf = (c) => {
  let g = Game.start(c.fen || undefined);
  for (const uci of c.moves) g = g.play({ from: uci.slice(0, 2), to: uci.slice(2, 4), promotion: uci[4] });
  return g;
};
const label = (c) => `${c.fen ? c.fen.split(" ")[0] : "start"} +${c.moves.length}`;

async function main() {
  // 1. Encoding: legal actions, observation, outcome.
  let n = 0;
  for (const c of cases.encoding) {
    const g = lineOf(c);
    const legal = g.legalMoves().map((m) => m.action);
    if (JSON.stringify(legal) !== JSON.stringify(c.legal)) { fail(`legal actions at ${label(c)}`); continue; }
    const obs = g.observation(), ones = new Set(c.ones);
    let bad = null;
    for (let i = 0; i < obs.length && !bad; i++) {
      const plane = i % 119;
      if (plane === 113 || plane === 118) {
        const want = c.scalars[plane === 113 ? 0 : 1];
        if (Math.abs(obs[i] - want) > 1e-6) bad = `plane ${plane}: ${obs[i]} != ${want}`;
      } else if ((obs[i] === 1) !== ones.has(i) || (obs[i] !== 0 && obs[i] !== 1)) {
        bad = `square ${Math.floor(i / 119)} plane ${plane}: ${obs[i]}`;
      }
    }
    if (bad) { fail(`observation at ${label(c)}: ${bad}`); continue; }
    const [over, value] = g.outcome();
    if (over !== c.over || value !== c.value) { fail(`outcome at ${label(c)}: ${over},${value} != ${c.over},${c.value}`); continue; }
    n++;
  }
  console.log(`encoding: ${n} of ${cases.encoding.length} positions match`);

  // 2. Networks (TensorFlow.js, CPU backend): logits and value against ws_zero.net.apply.
  if (cases.networks.length) {
    globalThis.tf = require(path.join(ROOT, "build/vendor/tf.min.js"));
    await globalThis.tf.setBackend("cpu");
    const { loadNetwork } = require(path.join(ROOT, "arena/js/network.js"));
    for (const { model, cases: ncases } of cases.networks) {
      const manifest = JSON.parse(fs.readFileSync(model + ".json", "utf8"));
      const buf = fs.readFileSync(model + ".bin");
      const net = loadNetwork(manifest, buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength));
      let worst = 0, worstValue = 0;
      for (const c of ncases) {
        const g = lineOf(c);
        const { logits, value } = await net.evaluate(g);
        c.legal.forEach((a, i) => { worst = Math.max(worst, Math.abs(logits[a] - c.logits[i])); });
        worstValue = Math.max(worstValue, Math.abs(value - c.value));
      }
      const ok = worst < 1e-3 && worstValue < 1e-4;
      if (!ok) fail(`network ${path.basename(model)}: logits off by ${worst}, value off by ${worstValue}`);
      else console.log(`network ${path.basename(model)}: ${ncases.length} positions, logits within ${worst.toExponential(1)}, value within ${worstValue.toExponential(1)}`);
      net.dispose();
    }
  }

  // 3. Search with the material counter: root visit counts against tests/reference_search.py, exactly.
  for (const c of cases.searches) {
    const g = lineOf(c);
    const result = await search(g, materialCounter, c.simulations);
    const got = {};
    for (const m of result.moves) if (m.visits) got[m.move.action] = m.visits;
    const want = Object.fromEntries(Object.entries(c.visits).map(([a, v]) => [a, v]));
    if (JSON.stringify(got) !== JSON.stringify(want)) fail(`search at ${label(c)}: ${JSON.stringify(got)} != ${JSON.stringify(want)}`);
    else console.log(`search at ${label(c)}: ${c.simulations} simulations, visit counts identical`);
  }
  console.log(failures ? `${failures} FAILED` : "all checks passed");
  process.exit(failures ? 1 : 0);
}
main().catch((e) => { console.error(e); process.exit(2); });
