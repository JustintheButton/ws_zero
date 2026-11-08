# Phase 3 gate report: network

**Date:** 2026-11-08. **Status: NOT PASSED.** The network is built and tested, throughput is measured on the Mac and
on six Modal GPU types, and the value head was amended. The overfit check still fails: when one small batch is
trained on over and over, a single training step eventually blows up and pins the value output at ±1. Under
realistic training (many positions, minibatches) the amended network learns; longer runs and a gradient cap are
being tested before Justin decides how to proceed.

## What was built

- `ws_zero.net`: the network from protocol §2 in plain JAX. The weights are a flat dict of named arrays (the names
  the WSP prompt will use): stem, 6 residual blocks, policy head (8 × 8 × 73 → 4,672 logits in action order),
  value head (tanh). 570,379 parameters. Also the random-arm initialization (He normal; zero biases; zero last layer
  of every residual branch and of both heads), masked softmax, the loss, and `value_head_health`.
- `ws_zero.train`: SGD with momentum 0.9 and weight decay 1e-4 on weights only (PyTorch style), and a jitted
  training step. The replay buffer and training loop are Phase 7.
- PyTorch removed; run logs record the JAX, jaxlib and pgx versions and the devices.
- Scripts: `gate_phase3.py` (overfit check), `check_value_head.py` (realistic check), `bench_phase3.py`
  (throughput), `modal_bench.py` (the benchmark on several Modal GPU types).

## Tests

84 unit tests pass. New ones include:
- parameter count, shapes, initialization statistics;
- a randomly initialized network gives exactly uniform move probabilities and value 0;
- the policy layout (square and move type → action number), checked against the Phase 2 move encoding;
- **hand-written networks**: three nonzero weights that make the network prefer double pawn pushes (both colors),
  and a few that make the value read whether a pawn stands on e2. A small version of Phase 6, confirming that the
  layout documented for the WSP prompt is the one the code uses. Deliberately breaking the layout makes them fail;
- the value conv passes negative values (the amendment), and `value_head_health` detects a constant, a saturated
  and a dead value head.

## Value head: found broken, amended

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

**Amended design, realistic check** (`phase-3-value-check.json`): all three seeds learn, held-out value error
0.52 → 0.20 / 0.20 / 0.17 after 900 steps, at most 1% of values saturated.

## Overfit check (gate)

Now run with spread-out move targets and realistic results (Justin's suggestion), as well as the original random
labels. 256 positions, one full batch, protocol optimizer, LR 0.003.

| Targets | Seed 0 | Seed 1 | Seed 2 |
|---|---|---|---|
| Realistic | fail | fail | fail |
| Random | fail | fail | fail |

Same pattern every time (`phase-3-overfit-{realistic,random}-seed{0,1,2}.json`): the value is fitted early (seed 0,
realistic: error 0.008 by step 150, under 0.007 until step 500) while the moves keep improving; then one step blows
up, the value output saturates at ±1 for every position, and the tanh passes almost no gradient, so it stays stuck.
Moves end up fitted (KL < 0.01). Gradient norms in this run: median 0.9, 90th percentile 4.8, 99th 17, max 63.5.

**Reading.** The underlying problem is an occasional explosive training step, which batch norm normally prevents.
The original head died from it; the amended head saturates. Repeating one small batch thousands of times provokes
it; the realistic check did not hit it within 900 steps. Running overnight: the realistic check for 3,000 steps,
and both checks with a gradient cap at norm 5 or 2 (above most steps, so only the rare large ones are cut).

## Throughput

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

1. **The explosive-step instability** (above): Justin's decision, informed by the overnight runs.
2. The game budget: the protocol fixes it from the cost per game measured in Phase 3; real self-play can first be
   measured in Phase 4. Justin's call. Compute available was raised to 11 Modal workspaces of $30 each (2026-11-08);
   a $30 hard limit per workspace is set by Justin in Modal, and the run launcher (to be built) will enforce it again.
3. Phase order: the Phase 7 smoke run is the real test of value-head health, and it currently comes after Claude
   writes weights (Phase 6). Running it first would avoid wasting Claude sessions if the network must change.
