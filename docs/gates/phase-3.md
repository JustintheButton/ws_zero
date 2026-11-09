# Phase 3 gate report: network

**Date:** 2026-11-08. **Status: PASSED, with batch norm restored.** The network is built and tested, and
throughput is measured on the Mac and on six Modal GPU types. Without batch norm (protocol v1.0) the value head
failed: first its single ReLU channel died, then, with that ReLU removed, rare explosive training steps pinned the
value at ±1 (one seed in three, for good, under realistic training). Justin restored batch norm, as in AlphaZero,
in every arm. With it the overfit gate passes on every seed with both kinds of targets, and the value head stays
healthy through 3,000 steps of realistic training on every seed, at LR 0.003. At the two largest D1 learning
rates (0.05 and 0.2) the value map's single ReLU channel can still shut off, mostly prevented by the protocol's
warmup; Justin added a pilot rule so that a learning rate that leaves the value head broken cannot be chosen.

## What was built

- `ws_zero.net`: the network from protocol §2 in plain JAX. The weights are a flat dict of named arrays: stem,
  6 residual blocks, policy head (8 × 8 × 73 → 4,672 logits in action order), value head (tanh), with batch norm
  after every conv except the policy output, as in AlphaZero. 571,276 trainable parameters plus 1,794 batch norm
  running statistics. `apply` (play and evaluation: running statistics) and `apply_train` (training: batch
  statistics, returns the new running statistics). Also the random-arm initialization (He normal; zero biases;
  identity batch norms; zero scale on the last batch norm of every residual branch and zero last layer of both
  heads, so the network starts as an identity stack with uniform policy and value 0), masked softmax, the loss,
  and `value_head_health`.
- `ws_zero.train`: SGD with momentum 0.9 and weight decay 1e-4 on conv and linear weights only (PyTorch style), and
  a jitted training step that stores the new running statistics and reports the gradient norm. The replay buffer
  and training loop are Phase 7.
- PyTorch removed; run logs record the JAX, jaxlib and pgx versions and the devices.
- Scripts: `gate_phase3.py` (overfit check), `check_value_head.py` (realistic check), `bench_phase3.py`
  (throughput), `modal_bench.py` (the benchmark on several Modal GPU types), `stability_phase3.py` (long runs
  with an optional gradient-norm cap).

## Tests

90 unit tests pass. New ones include:
- parameter count, shapes, initialization statistics;
- a randomly initialized network gives exactly uniform move probabilities and value 0;
- the policy layout (square and move type → action number), checked against the Phase 2 move encoding;
- **hand-written networks**: three nonzero weights that make the network prefer double pawn pushes (both colors),
  and a few that make the value read whether a pawn stands on e2. A small version of Phase 6, confirming that the
  layout documented for the WSP prompt is the one the code uses. Deliberately breaking the layout makes them fail;
- batch norm: training normalizes with the batch's statistics and moves the running ones by momentum 0.1; play
  uses the running ones; running statistics get no gradient and no weight decay, and the training step stores them;
- the value head is AlphaZero's (conv → batch norm → ReLU), and `value_head_health` detects a constant, a
  saturated and a dead value head, including a dead value map (the original failure).

The hand-written networks start from `blank_params` (every weight 0, every batch norm the identity).

## Without batch norm (protocol v1.0): value head broken

Everything in this section used the network without batch norm; the result files carry
`no-bn` or `original-head` in their names.

### First failure: the value ReLU died

**Original design** (protocol v1.0, AlphaZero's head with batch norm removed: 1×1 conv → ReLU → 256 → tanh). The
single ReLU channel can die: a large step pushes it negative on every square, after which it passes nothing and no
gradient reaches it again.

- Memorizing random labels (256 positions): moves memorized on every seed; value lost on 2 of 3 seeds
  (`phase-3-overfit-random-original-head-seed{0,1,2}.json`).
- **Realistic check** (`check_value_head.py`: 16,384 positions from random games, a soft move target that favours
  captures, a material-based result target, minibatches of 1,024, LR 0.003, scored on 2,048 held-out positions):
  the value head died in **all three seeds within 300 steps** and learned nothing (held-out error 0.52 → 0.50–0.52),
  while moves were learned normally (`phase-3-value-check-original-head.json`).
- Variants tried on the memorization test first (lower and higher LR, warmup, gradient clipping at norm 1.0, soft
  targets) did not fix it; clipping at 1.0 made it worse because most steps exceed that size.

**Amendment (Justin, 2026-11-08, before any results):** remove the ReLU after the 1×1 value conv. Same parameter
count, nothing else changes. Approved on the condition that every seed learns in the realistic check.

**Amended design, realistic check** (`phase-3-value-check-no-bn.json`): all three seeds learn, held-out value error
0.52 → 0.20 / 0.20 / 0.17 after 900 steps, at most 1% of values saturated.

### Second failure: the overfit check

Run with spread-out move targets and realistic results (Justin's suggestion), as well as the original random
labels. 256 positions, one full batch, protocol optimizer, LR 0.003.

| Targets | Seed 0 | Seed 1 | Seed 2 |
|---|---|---|---|
| Realistic | fail | fail | fail |
| Random | fail | fail | fail |

Same pattern every time (`phase-3-overfit-{realistic,random}-no-bn-seed{0,1,2}.json`): the value is fitted early (seed 0,
realistic: error 0.008 by step 150, under 0.007 until step 500) while the moves keep improving; then one step blows
up, the value output saturates at ±1 for every position, and the tanh passes almost no gradient, so it stays stuck.
Moves end up fitted (KL < 0.01). Gradient norms in this run: median 0.9, 90th percentile 4.8, 99th 17, max 63.5.

**Reading.** The underlying problem is an occasional explosive training step, which batch norm normally prevents.
The original head died from it; the amended head saturates. Repeating one small batch thousands of times provokes
it; the realistic check did not hit it within 900 steps, but it does over longer training (below).

### Overnight: longer training and a step cap

`stability_phase3.py` (`phase-3-stability.json`): both checks for 3,000 steps, LR 0.003, evaluated every 50
steps, with no cap (the protocol) or with gradient-norm clipping at 5 or 2. "Saturated" means at least half the
values pinned at ±1. Value error and move KL are on the 256 training positions (overfit) or the 2,048 held-out
positions (realistic).

| Test | Cap | Saturated | Value error (seeds 0 / 1 / 2) | Move KL | Overfit gate |
|---|---|---|---|---|---|
| Realistic | none | seed 0 from step 250, **never recovered**; seed 2 at step 250, recovered | 1.57 / 0.12 / 0.09 | 0.20–0.27 | |
| Realistic | 5 | never | 0.12 / 0.09 / 0.09 | 0.33–0.37 | |
| Overfit | 5 | never | 0.0001 / 0.0002 / 0.0002 | 0.009–0.010 | **pass 3/3** |
| Overfit | 2 | never | 0.014 / 0.002 / 0.0004 | 0.052–0.077 | fail 3/3 (moves too slow) |
| Overfit | none (earlier) | every seed, steps 400–550 | 1.56–1.63 | 0.008–0.009 | fail 3/3 |

1. **The problem is real, not an artifact of the overfit test.** Under realistic training without a cap, one seed
   in three died for good. Seed 0 had passed the 900-step check with the same starting weights and the same
   batches; the overnight script also measures each step's size, which most likely changes floating-point rounding
   slightly, and that was enough to send it the other way. Whether a run dies is a matter of chance, so the earlier
   3/3 did not establish that training is safe.
2. **A cap of 5 prevents it in both tests** and passes the overfit gate on every seed. The value is learned as well
   as by the healthy uncapped seeds.
3. **But a cap of 5 slows move learning in realistic training** (held-out KL 0.33–0.37 against 0.20–0.27). There
   the typical step is larger (median gradient norm 7–9 in the capped runs), so the cap trims most steps rather
   than only the rare huge ones. A cap of 2 is too tight even for the overfit gate.
4. These runs use LR 0.003, the smallest of the four D1 pilot values (0.2, 0.05, 0.0125, 0.003); the larger ones
   can only be less stable. D1's fallback, a cap of 1.0 only if every pilot setting diverges, is tighter than the
   cap of 2 that already slows learning.

## Batch norm restored: gate passed

**Decision (Justin, 2026-11-08, before any results):** restore batch norm in every arm, as in AlphaZero, with
AlphaZero's original value head (conv → batch norm → ReLU); options offered were an always-on gradient cap
(recommended), a cap of 5 now, batch norm, or no change. This departs from the checklist's "batch norm removed or
folded to identity" (protocol §10). Same tests, same seeds, same LR 0.003, no cap:

| Test | Seed 0 | Seed 1 | Seed 2 |
|---|---|---|---|
| Overfit gate, realistic targets | pass at step 1,100 | pass at 1,100 | pass at 1,150 |
| Overfit gate, random labels | pass at 600 | pass at 650 | pass at 575 |
| Realistic check, 3,000 steps | pass | pass | pass |

- **Overfit gate** (`phase-3-overfit-{realistic,random}-seed{0,1,2}.json`): at most 7% of values saturated with
  realistic targets and 21% with random labels at any check; value error 0.0002–0.0008 when passing.
- **Realistic check** (`phase-3-value-check.json`, held-out positions, checked every 100 steps): the value head
  healthy at every check on every seed (at most 11% saturated; value map at least 13% active; value layer at least
  48% active). Held-out value error 0.52 → 0.15 / 0.10 / 0.12 at 900 steps → **0.07 / 0.05 / 0.07** at 3,000,
  better than any run without batch norm (0.09–0.12 for the healthy ones).
- **But moves are learned more slowly at this LR.** Held-out move KL at 3,000 steps is 0.57–0.59, against 0.20–0.27
  without batch norm. The two are level up to about 900 steps (1.17–1.20 against 0.94–1.04); after that the network
  without batch norm speeds up. LR 0.003 is the smallest of the four D1 pilot values, and batch norm networks are
  normally trained at much larger ones (AlphaZero: 0.2).
- **Overfit gate run to 3,000 steps** (`stability_phase3.py`, `phase-3-stability-bn.json`; the gate script stops
  100 steps after passing): no saturation on any seed, value error 0.0000–0.0001, move KL 0.009–0.011 at the end.
  Gradient norms fall to almost nothing once the batch is fitted (median 0.0, maximum 1.0), where without batch norm
  they reached 33–65.

### The larger D1 learning rates

Realistic check, 1,000 steps each (`phase-3-lr-diagnostics.json`; scripts `check_value_head.py --lr`,
`value_relu_lr_phase3.py`, `warmup_phase3.py`). "Broken" means value map or value layer active on 1% of units or
fewer, or at least half the values saturated. Value error and move KL on the held-out positions at step 1,000.

| Value head | LR | Warmup | Seed | Broken? | Value error | Move KL |
|---|---|---|---|---|---|---|
| AlphaZero's | 0.003 | none | 0 | never | 0.14 | 1.13 |
| AlphaZero's | 0.0125 | none | 0 | never | 0.13 | 0.51 |
| AlphaZero's | 0.05 | none | 0 | **off from step 25, still off at 1,000** | 0.52 | 0.24 |
| AlphaZero's | 0.05 | none | 1 | off from step 25, recovered | 0.07 | 0.22 |
| AlphaZero's | 0.05 | none | 2 | off by step 100, recovered | 0.14 | 0.23 |
| AlphaZero's | 0.2 | none | 0 | **off by step 100, still off at 1,000** | 0.52 | 0.14 |
| AlphaZero's | 0.05 | 300 steps | 0 | never | 0.07 | 0.26 |
| AlphaZero's | 0.2 | 300 steps | 0 | off from step 100, recovered | 0.07 | 0.13 |
| No ReLU | 0.003 | none | 0 | never | 0.10 | 1.12 |
| No ReLU | 0.0125 | none | 0 | never | 0.04 | 0.50 |
| No ReLU | 0.05 | none | 0 | **saturated from step 100** | 1.57 | 0.23 |
| No ReLU | 0.2 | none | 0 | **saturated from step 100** | 1.57 | 0.13 |

1. **The shut-off is the original failure, now at large learning rates.** With one channel, a single large step can
   push batch norm's one scale and offset so that the value map is negative on every square; the ReLU then passes
   nothing and no gradient reaches it. It is real in training mode, not an artifact of the running statistics
   used in play (`warmup_phase3.py` measured both).
2. **With batch norm it can recover.** The rest of the network keeps changing (the trunk is shared with the move
   head), which can lift some squares above zero again; in three runs the value map came back and the value was
   then learned normally. Without batch norm it never recovered in these tests.
3. **Warmup mostly prevents it.** Without warmup, 2 of 4 runs at the two largest rates ended broken; with a
   300-step linear warmup (the protocol uses 1% or 3% of the budget), 0 of 2 did.
4. **Removing the ReLU does not help** at these rates: the value saturates at ±1 instead.
5. Moves are learned faster the larger the rate, so the network is otherwise stable at these rates.

**Decision (Justin, 2026-11-08):** keep the network and the learning-rate grid, and add a pilot rule to D1: a
learning rate whose value head is broken at the end of its pilot run counts as diverged and cannot be chosen
(protocol §10). "At the end" rather than "ever", since a shut-off can recover. Options offered: this rule
(recommended), no change, drop 0.2 from the grid, remove the ReLU.

## Throughput

The measurements below were made before batch norm was restored. In play, batch norm adds one multiply and one add
per value, which could be folded into the conv weights; Phase 4 measures real self-play with the final network.


Mac M4 Pro, CPU only (`phase-3-bench-mac.json`): network 19–22k evaluations/s; one search simulation (pgx step +
network) 10–16k/s; training 3.2 steps/s at batch 1,024. Estimate: 350–520 self-play games per hour for one run.

Modal (`phase-3-bench-modal.json`, 4 CPU cores and 16 GiB per container, list prices of 2026-11-08), one run per
GPU (100 games in flight, protocol D2), cost per 1,000 games from games averaging 340 plies to games reaching 512:

| GPU | $/hour | Games/hour | $ per 1,000 games |
|---|---|---|---|
| **L4** | 1.12 | 2,800–4,200 | **0.27–0.40** |
| A10 | 1.42 | 2,700–4,100 | 0.35–0.52 |
| L40S | 2.27 | 3,600–5,400 | 0.42–0.63 |
| T4 | 0.91 | 1,100–1,600 | 0.57–0.85 |
| A100-40GB | 2.42 | 2,500–3,800 | 0.63–0.95 |
| H100 | 4.27 | 3,600–5,500 | 0.78–1.17 |

With several runs sharing one GPU the big GPUs become cheapest (H100 at 6,400 games in flight: $0.10–0.16 per
1,000 games, 27,000–41,000 games/hour). Search bookkeeping is not included; Phase 4 measures real self-play.

**Consequence of D2.** One run gives the GPU only 100 positions per network call, and each 100-game chunk lasts as
long as its longest game. Packing several runs onto one GPU, or fixing the budget from Phase 4's real measurement,
are decisions for when the game budget is fixed.

## Open issues

1. **Written weights and batch norm** (decisions.md, open question 6): in training mode batch norm rescales each
   channel by the batch's statistics, so a network Claude writes would not compute what Claude wrote once training
   starts. Proposal: Claude writes the network without batch norm, and the harness sets each batch norm to the
   identity on sample positions before training. To be decided before Phase 6.
2. The game budget: the protocol fixes it from the cost per game measured in Phase 3; real self-play can first be
   measured in Phase 4. Justin's call. Compute available was raised to 11 Modal workspaces of $30 each (2026-11-08);
   a $30 hard limit per workspace is set by Justin in Modal, and the run launcher (to be built) will enforce it again.
3. Phase order: the Phase 7 smoke run is the real test of value-head health, and it currently comes after Claude
   writes weights (Phase 6). Running it first would avoid wasting Claude sessions if the network must change.
