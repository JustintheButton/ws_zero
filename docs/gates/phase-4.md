# Phase 4 gate report: search

**Date:** 2026-11-09. **Status: PASSED (Justin), 49 of 50 runs at 800 simulations.** The rule set before the run
asked for all 50. AlphaZero's search is built on mctx and checked against an independent plain-Python search, with
which it agrees visit for visit. With a dummy network that rates every move as equally likely and counts material,
the search finds every mate in one in every run and keeps its queen safe in 19 of 20 runs. The one miss is a
near-tie in the position where a knight attacks the queen. Along the way, the rule for untried moves was revised
(Justin), the whole project moved from 200 to 800 simulations per move (Justin), and a bug that made the search
dig a chain below finished games was fixed. The fix cut the 800-simulation gate from 116 minutes to 79 seconds and
left every result unchanged. After the gate, measuring real self-play found a second slowdown: every backup step
copied whole tables of the search tree. Our own backup replaced mctx's, with identical results; the gate now takes
15 seconds, and on one Modal L4 a move for 100 games at 800 simulations takes 0.92 seconds.

## What was built

- `ws_zero.search`, on mctx 0.0.71 (`muzero_policy`):
  - `make_root` and `make_recurrent_fn`, which connect mctx to the chess environment and a network. Values are
    for the side to move. A move scores reward + discount × (value of the next position), with discount −1
    because the opponent moves next (the sign flip). A move that ends the game gives the mover the result (+1 for
    checkmate, 0 for a draw) with discount 0. Illegal moves get probability exactly 0. Below a finished game every
    move is allowed, all equally likely (see "The chain below finished games").
  - `alphazero_qtransform`, AlphaZero's value term in place of mctx's default. A tried move scores its mean value
    Q on the −1…+1 scale. An untried legal move scores the position's own current value V(s). An illegal move
    scores −∞, because mctx excludes illegal moves only at the root.
  - `add_root_noise`: AlphaZero's root noise, Dirichlet(0.3) over the legal moves only, mixed in at 0.25.
    mctx's built-in noise is drawn over all 4,672 moves, so almost none of it would reach the ~30 legal ones; it
    is switched off.
  - `choose_moves` and `search_moves`. In self-play the move is drawn in proportion to visit counts for the
    first 30 plies, then the most-visited move is played. In evaluation the most-visited move is played from the
    first ply. Root noise is used in self-play only.
  - Constants from protocol §3: c(s) = log((N + 19652 + 1) / 19652) + 1.25, and 800 simulations per move.
  - `backward`: the backup after each simulation, our own, installed in place of mctx's (see "The full-table copy
    in every backup"). The module refuses to load with any mctx other than 0.0.71.
- `tests/reference_search.py`: a slow plain-Python AlphaZero search on python-chess positions, written
  independently of mctx, to check the real search against.
- `scripts/gate_phase4.py`: the gate.

Differences from the published pseudocode (protocol §3, and the `ws_zero.search` docstrings):
- mctx counts the root's own evaluation as a visit, so N(root) is one more than in the pseudocode. This only
  changes sqrt(N) in the root's exploration bonus.
- Three slips in the pseudocode are not copied:
  - its backup credits each node's value to the wrong player for selection;
  - it evaluates finished positions with the network;
  - it leaves its Dirichlet draws unnormalized.

## Tests

123 unit tests pass, 27 of them for the search (115 and 25 when the gate passed):
- **Root and recurrent function:** delivering mate rewards the mover +1 and ends the line; stalemate scores 0;
  stepping past the end of a game changes nothing; everything runs on a batch under jit.
- **Value term:** a tried move scores its mean value, an untried one the position's value, and an illegal one −∞.
  In a level position every root move is tried. A side that is ahead still tries every root move and finds the
  mate. The search never plays an illegal move below the root.
- **Sign flip at every level of the tree,** checked on a pawn ending searched at least 4 moves deep. This is the
  most common AlphaZero bug.
- **Exact agreement with the reference search:** identical visit counts for every root move after 100 simulations
  in 5 positions (start, Italian, Scholar's mate, mate in one, pawn ending).
- **Move choice and noise:** self-play samples in proportion to visits, and from ply 30 plays the most-visited
  move; evaluation plays the most-visited move from ply 1. Root noise is Dirichlet over the legal moves at 0.25,
  checked by its spread, and is used in self-play only. A finished game in a batch is harmless.
- **No chain below a finished game:** the tree never grows more than one move below one. This test failed before
  the fix.
- **The backup (added after the gate):** the compiled search holds no select over a whole tree table (it found 2
  before the fix), and 4 games searched together at 100 simulations, with backups of different lengths, build
  exactly mctx's tree, every number including the stored positions. Broken on purpose (finished games kept
  writing), the second test failed on 91 numbers.

## Gate

**Design (protocol §10).** A uniform-policy dummy network with value 0 cannot see that losing a queen is bad. The
gate therefore uses a dummy network that rates every legal move as equally likely and values a position at
tanh((own material − opponent material) / 4), with pawn 1, knight 3, bishop 3, rook 5 and queen 9. The search runs
as in evaluation games: no noise, most-visited move. There are 10 positions, each searched with 5 seeds, because
mctx breaks exact ties with tiny random noise and a uniform policy makes ties common.
- **Mate in one:** a run passes if the move played gives mate.
- **Hanging queen:** a run fails if, after the move, the side's queen can be taken by a cheaper piece, or taken
  at all while undefended.
- python-chess checks that every position is what it claims: a mate exists, and both a queen-hanging move and a
  safe move exist.

**Results, three runs:**

| Position | Untried = 0, 200 sims | Untried = V(s), 200 sims | Untried = V(s), 800 sims |
|---|---|---|---|
| Back rank (Ra8#) | 0/5 | 5/5 | 5/5 |
| Scholar's mate (Qxf7#) | 5/5 | 5/5 | 5/5 |
| Smothered mate (Nf7#) | 5/5 | 5/5 | 5/5 |
| Back rank, Black to move (Ra1#) | 1/5 | 5/5 | 5/5 |
| Queen and king (Qg8#) | 0/5 | 5/5 | 5/5 |
| Mate by promotion (e8=Q# or e8=R#) | 5/5 | 5/5 | 5/5 |
| Queen attacked by a pawn | 5/5 | 5/5 | 5/5 |
| Poisoned f7 pawn | 5/5 | 5/5 | 5/5 |
| Queen attacked by a knight | 2/5 | 0/5 | **4/5** |
| Queen attacked by a pawn, Black to move | 5/5 | 5/5 | 5/5 |
| **Total** | **33/50** | **45/50** | **49/50** |

**Run 1: untried moves scored 0 ("equal").** When the side to move was ahead, every move it had tried scored about
+0.85 from the material count alone, so any untried move looked worse. In the back-rank position the search spent
171 of 200 simulations on Ra2 and never tried Ra8#. Justin revised the rule: an untried move scores the position's
own value V(s), as in Leela Chess Zero (which also subtracts a small penalty) and mctx. In a level position this is
0, as before, so the random arm is not handicapped.

**Run 2: untried moves scored V(s), 200 simulations.** Every mate was found in every run. The knight position failed
in every run. To see that a quiet move such as a3 loses the queen, the search must try that move and then find
Black's knight capture among about 35 replies, for each of White's 38 moves: roughly 1,300 lines. Spread over
200 simulations, about five per move, many moves looked equally fine, and the tie went to a3. I recommended
keeping 200 and replacing the position. Justin moved the whole project to AlphaZero's 800 (protocol §10).

**Run 3: 800 simulations.** In the knight position the three most-visited moves were queen retreats (Qc3 34,
Qb3 34, Qg3 33). Four runs played a safe queen move (Qb3 twice, Qh3, Qh5); one played b3. Justin judged the miss a
near-tie, not a bug: every mate was found in every run, the other queen positions passed 15/15, and the search
agrees exactly with the reference. He passed the gate at 49/50. The gate script keeps its original rule and still
prints "not passed"; the verdict is recorded in decisions.md and the checklist.

Results: `phase-4-gate.json` (800 simulations, after the fix).

## The chain below finished games

The 800-simulation run took 116 minutes, against 2.3 minutes at 200. Timing one search of the 10 positions:

| Simulations | Chess search | Growth | mctx alone, same size |
|---|---|---|---|
| 100 | 4.3 s | | 0.16 s |
| 200 | 26.8 s | ×6.2 | 0.45 s |
| 400 | 187 s | ×7.0 | 1.4 s |
| 800 | about 23 min | | not run |

("mctx alone" uses a do-nothing game with 4,672 moves: no chess, no network.)

**Cause.** Below a finished game, every move was masked as illegal, so each return visit to a mate took the same
slot one move deeper. At 200 simulations, 189 of the 200 nodes in the smothered-mate tree hung in a chain 190
moves deep below Nf7#. In the queen positions, which have no mate in reach, trees stayed 2–5 moves deep. Every
simulation walked the chain, and the whole batch waits for its deepest game.

**Fix.** Below a finished game every move is allowed, all equally likely, so return visits spread out one move
below it. Nothing there counts (discount 0), so results cannot change. The gate rerun at 800 gave identical moves,
visit counts and values in **79 seconds**. In real self-play the chain would have formed whenever the search saw a
mate or a draw coming, stalling all games in the batch.

## The full-table copy in every backup

The first real self-play measurement (100 games, 800 simulations, one Modal L4) did not finish a single move in
its 20-minute limit (about $0.37 spent). On the Mac, one move for every game in the batch took:

| Games × simulations | Simulations per move | Time per move |
|---|---|---|
| 100 × 50 | 5,000 | 1.8 s |
| 100 × 200 | 20,000 | 10.5 s |
| 25 × 800 | 20,000 | 33.7 s |

The last two do the same number of simulations on trees of the same total size, yet 25 × 800 took over 3 times
as long: the cost of a simulation grew with the tree.

**Cause.** After each simulation the backup climbs from the new node to the root, updating visit counts and mean
values. mctx writes this loop for one game and batches it over the games with vmap. The games climb paths of
different lengths, so the batched loop runs until the longest is done, and at every step chooses between the new
tree and the old one, game by game, over the whole of two tables: the visit counts and values of all 4,672 moves
at every node (1.5 GB each at 100 games × 800). A move therefore cost about games × simulations² × 4,672. The
compiled program showed exactly two such whole-table selects, inside the batched backup loop.

**Fix.** `ws_zero.search.backward` runs one loop until every game reaches its root and writes one entry per game
per step, with mctx's arithmetic. The tree is identical to mctx's (tested, above). The gate rerun gave identical
moves, visit counts and values in **15 seconds** (79 before).

## Throughput

Measured on one Modal L4 after the fix (`phase-4-selfplay-modal.json`): 100 games × 800 simulations from the start
position with an untrained network, root noise and sampled moves, over 40 moves.

| | |
|---|---|
| Compiling | 14 s |
| One move for all 100 games | 0.92 s (0.90–1.01) |
| One training step (batch 1,024) | 0.024 s, under 1% of the time |
| Games per hour | 758 (chunks of 512 plies) to 1,137 (chunks of 340 plies) |
| Cost per 1,000 games | $1.47 to $0.98 |

Phase 3 measured the network and the chess step alone at 1.23 ms per simulation for 100 games on an L4, or 0.98 s
for 800: the search's own bookkeeping now costs almost nothing, and the Phase 3 estimate (710–1,070 games per hour,
$1.05–1.58 per 1,000) holds. mctx still keeps statistics for all 4,672 moves at every node, so the search tree for
100 games is 9.9 GB at 800 simulations; the L4 has 23 GB.

**This is a best case** (found in the code review of 9 Nov). The measurement used an untrained network, whose
look-aheads spread evenly, so its trees are only 2–3 moves deep. Every look-ahead walks down its game's tree, and the
whole batch waits for the deepest line. A trained network concentrates on a few moves and digs deeper. On the Mac
(32 games × 200 simulations, the real network's cost, move probabilities set by hand to control the depth):

| Deepest line | One move for all 32 games |
|---|---|
| 2 moves | 1.5 s |
| 5 moves | 2.1 s |
| 12 moves | 3.1 s |
| 13 moves, more look-aheads going deep | 3.8 s |

How much of this carries over to a GPU is not measured: there the network is a smaller share of the time, so the
walk down the tree may weigh more or less. Real training games will cost more than the figures above, possibly up
to about twice as much.

## Open issues

1. **Game budget (D7).** The measured cost is about 7% below the Phase 3 estimate, but it is a best case (see
   Throughput): the speed must be measured again with deep trees or a trained network before the budget is fixed.
   On the best-case figure (not recomputed): at 800 simulations, roughly 7,000–10,000 games per run within the $192 cap, under the 15,000 line below which Justin's PC runs seeds 4–5. Even all 11 workspaces ($330) give only
   about 9,500–14,000. A Stanford cluster may become available. To settle before Phase 7.
2. **Search memory:** 9.9 GB per 100 games limits packing several runs onto one GPU (Phase 3 found packing the
   cheapest option on large GPUs). Keeping statistics for legal moves only (at most 218) would shrink the tree about
   20 times. The Mac has 24 GB.
3. **Carried over from Phase 3:** how WSP-written weights meet batch norm (before Phase 6), and whether the Phase
   7 smoke run should come before Phase 6.
4. Six old lint warnings in Phase 3 files (`check_value_head.py`, `gate_phase3.py`, `test_train.py`).
