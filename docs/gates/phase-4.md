# Phase 4 gate report: search

**Date:** 2026-11-09. **Status: PASSED (Justin), 49 of 50 runs at 800 simulations.** The rule set before the run
asked for all 50. AlphaZero's search is built on mctx and checked against an independent plain-Python search, with
which it agrees visit for visit. With a dummy network that rates every move as equally likely and counts material,
the search finds every mate in one in every run and keeps its queen safe in 19 of 20 runs. The one miss is a
near-tie in the position where a knight attacks the queen. Along the way, the rule for untried moves was revised
(Justin), the whole project moved from 200 to 800 simulations per move (Justin), and a bug that made the search
dig a chain below finished games was fixed. The fix cut the 800-simulation gate from 116 minutes to 79 seconds and
left every result unchanged.

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

115 unit tests pass, 25 of them for the search:
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

## Throughput

Not yet measured for real self-play. Phase 3's estimate (Modal L4, one run of 100 games per GPU) scales with
simulations: at 800, about 710–1,070 games per hour and $1.05–1.58 per 1,000 games, against 2,800–4,200 and
$0.27–0.40 at 200. That estimate leaves out the search's own overhead, which the next step measures. mctx keeps
statistics for all 4,672 moves at every node, so the search tree for 100 games is 9.9 GB at 800 simulations (2.5
GB at 200).

## Open issues

1. **Real self-play speed** (the next Phase 4 step) sets the game budget (D7). By the Phase 3 estimate, 800
   simulations give roughly 7,000–10,000 games per run within the $192 cap, under the 15,000 line below which
   Justin's PC runs seeds 4–5. Even all 11 workspaces ($330) give only about 9,500–14,000.
2. **Search memory:** 9.9 GB per 100 games limits packing several runs onto one GPU (Phase 3 found packing the
   cheapest option on large GPUs). The Mac has 26 GB.
3. **Carried over from Phase 3:** how WSP-written weights meet batch norm (before Phase 6), and whether the Phase
   7 smoke run should come before Phase 6.
4. Six old lint warnings in Phase 3 files (`check_value_head.py`, `gate_phase3.py`, `test_train.py`).
