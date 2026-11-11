# Phase 5 gate report: evaluation harness

**Date:** 2026-11-11 (runs on 2026-11-09). **Status: PASSED (Justin).** A player's strength is measured by playing
a fixed ladder of 12 opponents from 50 fixed openings, both colours, and fitting all results to one rating scale
with the random mover at 0. The gate entered the same player twice under two names, differing only in the seed of
its search's tie-breaking. For both players tested, the two copies rate as equal: the untrained network's copies
are 3 points apart (95% range −34 to +25) and the material-count stand-in's are 10 apart (−55 to +35). One
evaluation of 1,200 games takes 21–33 minutes on one Modal L4 and costs $0.47–0.72. Three checklist items are not
built: the tablebases, the two test-position sets, and the policy and value measures computed on them. They were
postponed to Phase 6, where they are first used (Justin, 2026-11-09).

## What was built

- **Stockfish 19, pinned** (`scripts/install_stockfish.py`, `ws_zero.stockfish`). Installed from the official
  release files and checked against their SHA-256; the engine's `bench` number must be 2497913, and it must replay
  a pinned set of moves and a pinned 40-ply game. One thread, 16 MB of hash, and the hash is cleared before every
  move, so a full-strength move depends only on the game so far. The Mac and Modal play the same moves.
  - `Stockfish(nodes)`: full strength at a fixed node count. `Stockfish(nodes, elo=1320)`: strength-limited.
  - `EnginePool`: several engines behind one player, for games played side by side.
  - `Analyst`: the same engine as a judge of positions at a fixed depth.
- **Weak opponents** (`ws_zero.players`): the random mover, the material grabber, and `Diluted`, which plays a
  random legal move with a set probability and otherwise asks another player. All seeded by game number.
  `play_game` plays one game under the project's rules (checkmate, stalemate, insufficient material, 100 plies
  without progress, automatic threefold repetition, 512-ply cap) and records how it ended.
- **The ladder** (`ws_zero.ladder`, protocol §8.2 as amended 2026-11-09), 12 opponents:
  random, grabber, the 1-node Stockfish with 75%, 50% and 25% random moves, Stockfish limited to 1320 at 10,000
  nodes, and Stockfish at 1, 100, 300, 1,000, 3,000 and 10,000 nodes.
- **Opening suites** (`scripts/make_openings.py`, `data/openings.json`, `ws_zero.openings`): random 8-ply lines kept
  only if Stockfish at depth 20 rates them within ±50 centipawns. 150 of 550 candidates were kept: 50 for checkpoint
  evaluation and a separate 100 for the GO / NO-GO test.
- **Match runner for our agent** (`ws_zero.evaluate`): all games of a match are searched together in one batch on
  the GPU, 800 simulations, no root noise, most-visited move from the first ply. Each opening is played once with
  each colour. The runner stops if pgx and the python-chess reference ever disagree about a game's end.
  With 0 simulations the agent plays the raw policy's top legal move (the protocol's secondary agent).
- **Ratings** (`ws_zero.ratings`): a Bradley–Terry fit of all games together, the first player at 0, and 95%
  ranges from refits on games redrawn with replacement within each pairing.
- **Scripts:** `ladder_matches.py` (round robin of the ladder), `evaluate_agent.py` (one agent against the whole
  ladder), `gate_phase5.py` (the gate), and the Modal launchers `modal_stockfish_job.py` (processor cores) and
  `modal_evaluate.py` (one L4 and 8 cores per evaluation). Matches run on Modal, not on the Mac.

## Tests

169 unit tests pass, 45 of them for this phase:
- **Stockfish (13):** the same position and node count always give the same move; a whole game replays move for
  move; a move depends only on the game so far, not on what the engine searched before; a pool of engines gives
  one engine's moves from many threads; a mate in one is played; an engine that is not Stockfish 19, or that plays
  other moves than the pinned ones, is refused.
- **Installer (3):** a download with the wrong checksum is refused and not kept.
- **Players and games (15):** the random mover is uniform and repeatable; the grabber takes the most valuable
  piece, counts en passant, and breaks ties at random; dilution asks its base player the right share of the time;
  games end by each rule with the right result; an illegal move stops the game.
- **Openings (4):** candidate lines are 8 legal plies, never finished, never the same position twice; the committed
  suites are 50 and 100 different balanced openings.
- **Match runner (6):** every opening is played once with each colour; games are legal and end as the rules say;
  the same seed replays the same games and another seed does not; with no search the agent plays its policy's top
  legal move.
- **Ratings (4):** a 75% score is 191 points; ratings chain through a middle player; two players with the same
  results are not told apart; a player that loses every game still gets a finite rating.

## Calibration of the ladder

All 12 opponents played each other on the 50 checkpoint openings, both colours: 66 pairings × 100 games
(`docs/gates/phase-5-calibration.json`, every game in `phase-5-calibration-games.json.gz`). 6,137 of the 6,600 games
ended in checkmate and 8 at the ply cap.

| Opponent | Rating, ladder alone | Score against the rung below |
|---|---|---|
| Random mover | 0 | |
| Material grabber | 248 | (out of order, see below) |
| 1-node Stockfish, 75% random | 191 | 84.5% against random |
| 1-node Stockfish, 50% random | 371 | 85% |
| 1-node Stockfish, 25% random | 635 | 85% |
| Stockfish limited to 1320, 10,000 nodes | 905 | 82.5% |
| Stockfish, 1 node | 1,456 | **98%** |
| Stockfish, 100 nodes | 1,569 | 64.5% |
| Stockfish, 300 nodes | 1,705 | 68.5% |
| Stockfish, 1,000 nodes | 2,103 | 89% |
| Stockfish, 3,000 nodes | 2,556 | 93% |
| Stockfish, 10,000 nodes | 2,955 | 90.5% |

What the calibration shows:
- **The lower rungs step as designed**, 82–85% each, up to the 1320 opponent.
- **There is a hole of about 550 points between the 1320 opponent and the 1-node Stockfish** (98%). In the first
  look on random openings the same step measured 320–350 points.
- **The upper steps are larger than on random openings:** 400–450 points each from 300 to 10,000 nodes, against
  230–275 before. From balanced openings the stronger engine wins almost every game. The whole span from 1 node to
  10,000 nodes is about 1,500 points, against about 1,040 before.
- **31 of the 66 pairings ended 100 to 0.** Rungs far apart tell the fit nothing about their distance, so the
  top of the ladder is placed only by the chain of neighbouring rungs.
- **The grabber does not fit on one scale.** It draws 84 games in 100 against both the 75% and the 50% rung by
  trading pieces off, and scores under 20% against the 25% rung. Its rating depends on who else is in the fit.

## Gate

**Design (protocol §10; Justin, 2026-11-09).** Two players, each entered twice: the untrained network
(`untrained`: standard random initialization, so a uniform policy and value 0) and the Phase 4 stand-in
(`material`: uniform policy, value from counting material). Both use the full search at 800 simulations. The two
copies of a player have the same weights and differ only in the seed of the search's tie-breaking. Each copy plays
all 12 opponents on the 50 checkpoint openings with both colours: 1,200 games. All four evaluations are fitted
together with the calibration games. The gate passes if, for each player, the 95% range of the gap between its two
copies includes 0 (500 refits on redrawn games).

**Result** (`docs/gates/phase-5-gate.json`, `uv run python scripts/gate_phase5.py`):

| Player | Copy 1 | Copy 2 | Gap | 95% range of the gap | |
|---|---|---|---|---|---|
| Untrained network | 114 | 117 | −3 | −34 to +25 | pass |
| Material stand-in | 596 | 606 | −10 | −55 to +35 | pass |

Score against each opponent (copy 1 / copy 2):

| Opponent | Untrained network | Material stand-in |
|---|---|---|
| Random mover | 74.5% / 73% | 88% / 89% |
| Material grabber | 48% / 46% | 88.5% / 85.5% |
| 75% random | 28.5% / 35% | 87% / 85% |
| 50% random | 10% / 7% | 82% / 86% |
| 25% random | 2% / 3% | 64% / 64.5% |
| Stockfish limited to 1320 | 0% / 1% | 14% / 18% |
| Stockfish, 1 node | 0.5% / 0% | 1.5% / 0.5% |
| Stockfish, 100 nodes | 0% / 0% | 0% / 1% |
| Stockfish, 300 nodes and above | 0% / 0% | 0% / 0% |

All ratings in the joint fit, random mover = 0, with 95% ranges:

| Player | Rating | 95% range |
|---|---|---|
| Stockfish, 10,000 nodes | 2,918 | 2,792 to 3,113 |
| Stockfish, 3,000 nodes | 2,520 | 2,388 to 2,700 |
| Stockfish, 1,000 nodes | 2,069 | 1,969 to 2,205 |
| Stockfish, 300 nodes | 1,673 | 1,587 to 1,776 |
| Stockfish, 100 nodes | 1,536 | 1,455 to 1,637 |
| Stockfish, 1 node | 1,423 | 1,340 to 1,532 |
| Stockfish limited to 1320 | 884 | 843 to 930 |
| Material stand-in, copy 2 | 606 | 565 to 649 |
| Material stand-in, copy 1 | 596 | 558 to 639 |
| 1-node Stockfish, 25% random | 579 | 547 to 613 |
| 1-node Stockfish, 50% random | 375 | 351 to 406 |
| 1-node Stockfish, 75% random | 215 | 192 to 240 |
| Material grabber | 214 | 194 to 241 |
| Untrained network, copy 2 | 117 | 94 to 144 |
| Untrained network, copy 1 | 114 | 90 to 138 |
| Random mover | 0 | fixed |

**What the gate also showed:**
- **The resolution of one evaluation.** The gap between two copies is known to within about 30–45 points. Two
  checkpoints closer than that look the same after 1,200 games each.
- **An unplanned repeat.** The second copy of the stand-in was cut off after 7 opponents (the Mac lost its
  connection to Modal) and run again. Against the six of those opponents that do not use their own random numbers,
  the second run repeated the first exactly: the same wins, draws and losses. Against the strength-limited
  Stockfish it scored 13% the first time and 18% the second. That opponent chooses its errors with random numbers
  that UCI gives no way to seed, so its games never repeat.
- **The ladder's ratings move when players are added to the fit.** The 25% rung is 635 with the ladder alone, 690
  with the untrained copies added, and 579 with all four. The cause is the same as the grabber's: among weak
  players many games are drawn at the ply cap or by bare kings, and a single number per player cannot hold all
  the pairings.

## Throughput and cost

| Run | Machine | Games | Time | Cost |
|---|---|---|---|---|
| Ladder calibration | Modal, 16 cores | 6,600 | 401 s | $0.09 |
| Untrained network, copy 1 | Modal L4 + 8 cores | 1,200 | 1,961 s | $0.72 |
| Untrained network, copy 2 | Modal L4 + 8 cores | 1,200 | 1,843 s | $0.67 |
| Material stand-in, copy 1 | Modal L4 + 8 cores | 1,200 | 1,296 s | $0.48 |
| Material stand-in, copy 2 | Modal L4 + 8 cores | 1,200 | 1,271 s | $0.47 |

- The whole gate cost about $2.90, including about $0.45 for the run that was cut off.
- Most of an evaluation's time goes to the weak opponents, where games run long: 129–283 s for each of the five
  weakest, 27–59 s for each of the three strongest. An untrained network cannot finish a won game, so 22 of its
  100 games against the random mover ran to the 512-ply cap.
- At $0.47–0.72, the plan's roughly 375 checkpoint evaluations would cost about $190–260. A trained network
  should finish weak opponents faster and last longer against strong ones, so the figure may move either way.

## Open issues

Decisions for Justin, none of them blocking Phase 6:
1. **Freeze the ladder or refit it with every checkpoint?** The joint fit moves the ladder's own ratings by tens of
   points (see above). Protocol §8.4 says "from all games jointly".
2. **The evaluation budget.** About $190–260 for evaluation alone, next to the open training budget (D7; the Phase
   4 report). To settle before Phase 7.
3. **The hole between the 1320 opponent and the 1-node Stockfish** (about 550 points) and the large steps above
   300 nodes. They matter for where the finish line (D8) can be put and how finely progress shows on the way up.
4. **Ordo.** Protocol §8.4 names Ordo; this phase used the project's own fit (`ws_zero.ratings`). To cross-check
   against Ordo, or to amend the protocol.

Postponed to Phase 6 (Justin, 2026-11-09), not dropped; the protocol is unchanged:
- Syzygy 3–5 piece tablebases.
- The two fixed test-position sets (middlegame positions labelled by Stockfish, endgame positions by tablebases).
- Policy top-1 agreement with Stockfish, and value accuracy and calibration, which are computed on those sets.

Known limits:
- The strength-limited Stockfish's games cannot be repeated (no seed over UCI).
- A Modal run started from the Mac dies if the Mac sleeps. Training-length runs need Modal's detached mode, with
  results stored on Modal (Phase 7).
- The secondary agent (raw policy, no search) is built and tested but has not been run against the ladder.
