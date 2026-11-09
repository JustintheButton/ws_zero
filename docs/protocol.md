# Pre-registered protocol

**Status: v1.0, approved by Justin on 2026-11-08.** Items marked **[D#]** record Justin's decisions (the reasoning
for each is in [decisions.md](decisions.md) and the session log in [logs/](logs)). From now on this file changes
only through a dated deviation in [decisions.md](decisions.md). Items marked *(fixed after Phase 3)* are set by the
rule stated here, using measured throughput, before any experiment runs.

---

## 1. Research question and hypotheses

**Question.** Does initializing AlphaZero's network with weights written directly by an LLM (Weight-Space
Programming, WSP) make early training more sample-efficient than standard random initialization?

- **H1 (primary).** WSP-initialized runs reach the strength threshold (§8.4) in fewer self-play games than
  random-initialized runs.
- **H2.** WSP-initialized runs reach the threshold in fewer self-play games than distillation-initialized runs, i.e.
  any benefit comes from writing weights, not merely from the LLM's chess knowledge.
- **H3 (descriptive).** The advantage, if any, washes out: the WSP and random learning curves converge within the
  game budget.
- **H4 (descriptive).** The LLM-written features persist through training (§9.3).

**Model note.** The WSP results we build on were obtained with GPT-6 Astra. Whether Claude Opus 5.5 can write
useful weights is itself untested; Phase 6 measures it before any training.

## 2. Architecture (identical in every arm)

| Part | Specification |
|---|---|
| Input | 119 × 8 × 8 planes (AlphaZero encoding), board flipped so the side to move is always at the bottom |
| Stem | 3×3 conv, 119 → 64 filters, batch norm, ReLU |
| Body | 6 residual blocks: 3×3 conv (64) → batch norm → ReLU → 3×3 conv (64) → batch norm → add skip → ReLU |
| Policy head | 3×3 conv (64) → batch norm → ReLU → 1×1 conv (73, with bias) → 4,672 logits; illegal moves masked before softmax |
| Value head | 1×1 conv (1) → batch norm → ReLU → linear 64 → 256 → ReLU → linear 256 → 1 → tanh (AlphaZero's) |
| Normalization | batch norm after every conv except the policy head's final 1×1 conv, as in AlphaZero, in every arm; convs followed by batch norm have no bias. Training normalizes with the batch's statistics; self-play and evaluation use running statistics (momentum 0.1, ε = 10⁻⁵, PyTorch's defaults) *(amended 2026-11-08: v1.0 had no batch norm; see §10)* |
| Size | 571,276 trainable parameters, plus 1,794 batch norm running statistics |

The value is from the side to move's perspective, in [−1, 1].

## 3. Search

- PUCT with c(s) = log((N(s) + 19652 + 1) / 19652) + 1.25; 200 simulations per move.
- Root Dirichlet noise α = 0.3, ε = 0.25 in self-play only.
- Self-play move choice: sample proportional to visit counts for the first 30 plies, then the most-visited move.
- A new tree for every move (no tree reuse); no resignation; draw adjudicated at 512 plies.
- Network evaluations are batched across parallel games (one leaf per game per batch), so no virtual loss is needed.
- First-play urgency (the value of unvisited children) follows the published AlphaZero pseudocode; checked and
  recorded in Phase 4. *Recorded 2026-11-08 (Justin):* the pseudocode scores an unvisited child 0, on a scale where
  its uniform network's value is 0.5 (0 = loss), so read literally an untried move counts as a loss; with a uniform
  policy the search would then not try a second root move within 200 simulations. Untried moves therefore score
  **0 on the −1…+1 value scale ("equal")**, the AlphaGo Zero paper's rule (Q(s, a) = 0 initially) and the
  pseudocode's 0 read on the AlphaZero paper's −1…+1 scale. Two slips in the pseudocode are not copied: its backup
  credits each node's value to the wrong player for selection (the search uses the sign flip), and it evaluates
  finished positions with the network (the search uses the game's result).

## 4. Training (identical in every arm)

- **Loss:** (z − v)² − πᵀ log p, plus L2 on weights via SGD `weight_decay = 1e-4`. This equals the
  pseudocode's `1e-4 · l2_loss(weights)`; it is not added a second time. Biases and batch norm scales and
  offsets are not decayed.
- **Targets:** π = root visit-count distribution; z = final game result from the side to move's perspective
  (+1 / 0 / −1).
- **Optimizer:** SGD, momentum 0.9.
- **Learning-rate schedule:** four values in ratio 1 : 0.1 : 0.01 : 0.001 (AlphaZero: 0.2 → 0.02 → 0.002 → 0.0002).
  Drops after 1/7, 3/7 and 5/7 of the self-play game budget (AlphaZero's 100k / 300k / 500k of 700k steps,
  scaled; defined by games because under D2 the total number of steps is not known in advance). A linear warmup
  from 0 to the first value spans the training steps of the first part of the game budget (D1).
- **[D1] Base learning rate, warmup and stability.** AlphaZero's 0.2 was tuned *with* batch norm, which v1.0
  removed (restored 2026-11-08, §10; the pilot and its fallback are unchanged). **Decided (Justin, 2026-11-08):** a short pilot on the **random arm only** (pilot seeds are
  never reused; pilot results are excluded from all analyses) chooses
  - the base LR from {0.2, 0.05, 0.0125, 0.003} (each ÷4), and
  - the warmup length from {1%, 3%} of the game budget,

  by the highest Elo at the end of a pilot budget of 20% of the main budget. Pilot runs follow the full-budget
  schedule and stop at 20%, so the winning pilot can be continued as the D8 calibration run. A pilot setting whose
  value head is broken at the end of its pilot run (`ws_zero.net.value_head_health`: value map or value layer active
  on 1% of units or fewer, or at least half the values saturated) counts as diverged and cannot be chosen
  *(added 2026-11-08, §10)*. If every pilot setting diverges,
  gradient-norm clipping (threshold 1.0) is added, for every arm, and the pilot is repeated. Tuning on the random
  arm only biases the comparison *against* WSP, which makes a positive result more convincing.
- **[D2] Loop shape and data reuse.** **Decided (Justin, 2026-11-08):** **synchronous chunks**: play 100 self-play games with the
  current network, then train for S steps, then repeat; no gating. S is set so each new position is sampled on
  average 4 times while in the replay window: S = 4 × (positions generated in the chunk) / batch size. Batch size
  1,024. This keeps AlphaZero's "continuous updates, no gating" at a small delay and, unlike asynchronous training,
  makes every run reproducible from its seed.
- **Replay window:** the most recent W games, sampled uniformly by position. W *(fixed after Phase 3)* = the
  larger of 2,500 games and 12.5% of the game budget.
- **Initialization of the random arm:** He (Kaiming) normal for all conv and linear weights, zero biases, batch
  norm scale 1 and offset 0 (running mean 0, variance 1); the scale of the second batch norm of each residual block,
  the policy head's final 1×1 conv and the value head's final linear layer are zero-initialized. The network
  therefore starts as an identity-like stack with a uniform policy and value 0 (zero-initializing the last batch
  norm scale of each residual branch is the standard way, Goyal et al. 2017; v1.0 zeroed the second conv instead,
  which suits a network without batch norm).
- **Divergence:** a run whose loss becomes NaN or infinite is recorded as diverged; it is not restarted. A run
  interrupted by an infrastructure failure (crash, power loss) resumes from its last checkpoint, and this is
  recorded.
- **Reproducibility:** every run is seeded (`seed_everything`, worker seeds via `derive_seed`). Bitwise
  reproducibility on the GPU is not guaranteed.

## 5. Arms

All arms share every setting above; only the initial weights differ.

1. **Random:** §4 initialization, a fresh draw per seed.
2. **WSP:** a network written by Claude Opus 5.5 in Phase 6 (budget 50 passes).
   **[D3] Which written networks.** **Decided (Justin, 2026-11-08):** **one different written network per seed**:
   seed k starts from budget-50 attempt k (k = 1–5), in the order the attempts were run (fixed before any
   evaluation). The
   result then answers "does the WSP *procedure* help", including its variance between attempts, rather than "does
   one lucky network help". A possible follow-up, outside this protocol, is every written network × 3 seeds
   (see [decisions.md](decisions.md#follow-up-ideas)).
3. **Distillation:** a network pretrained by supervised imitation of a Python chess heuristic written by the same
   LLM. **[D4] "Matched effort".** **Decided (Justin, 2026-11-08):**
   - **LLM side:** the same harness, model, effort level and limits as a budget-50 WSP session (120 Python runs,
     50 passes, where a games pass plays the *heuristic* against the practice opponents). The heuristic receives a
     python-chess board and must return a score per legal move and a value in [−1, 1]. It may not search: no
     lookahead beyond the current move, no engine, no learned components.
   - **Imitation:** 1,000,000 positions from heuristic-vs-heuristic games (moves sampled from its scores), with
     targets = its move distribution and value. Training uses §4's optimizer, loss and batch size at the D1 base learning
     rate, with a linear warmup over the first 3% of its steps, for a fixed 2 epochs; then self-play training starts with the same schedule as
     every other arm. The imitation compute is reported separately and does not count toward
     the self-play game axis (WSP's writing session doesn't count toward it either).
   - **One heuristic per seed** (five attempts), as in D3.
   - **Fidelity check:** after imitation, report the network's top-1 agreement with its heuristic and the error of
     its value against the heuristic's, on held-out positions. A poor copy is reported as a limitation of the
     control.

## 6. Phase 6 pilot: writing weights (WSP)

**Harness** (modelled on Experiment 6 of the WSP logs):
- Each turn is one call of Claude Code in non-interactive mode, resuming one session per attempt. All built-in
  tools are disabled; the system prompt is one neutral line. Each turn's event stream is audited for zero tool
  use.
- Claude ends each reply with exactly one fenced block: ```` ```python ```` (a sandbox run), ```` ```games ```` or
  ```` ```positions ```` (a pass). The harness runs it and sends back the output.
- **Sandbox** (bubblewrap under WSL2): no network, no GPU, a `/work` folder kept between runs; numpy, scipy,
  scikit-learn, PyTorch (CPU), pandas, sympy, networkx. **No chess library and no engine.** 900 s per run, output
  cut at 20,000 characters, at most 120 runs. A prelude provides `load` / `save` for the network's tensors (all
  zero at the start) and index helpers for input planes and policy moves.
- **Prompt sections:** task, rules of chess, the network (architecture, tensor names and shapes, input planes,
  4,672-move encoding, value semantics), how it will be measured, what it has, rules, how to finish. The finish
  must include a machine-readable map of which channels compute which features (used in §9.3).
- **Rules given to the model:** not allowed: (a) training or fitting weights to data or labels (no backward
  passes, autograd, optimizers, regression or least squares on examples); (b) search of any kind (no game-tree
  search, minimax, MCTS or rollouts, to choose moves or to make targets); (c) running the network outside the
  passes; (d) internet or engine access. Allowed: writing weights from its own design (formulas, loops, linear
  algebra on the weights), its own chess-rule code that neither runs the network nor searches, and analysis of
  pass results.
- **Passes:** a *games pass* plays up to 128 games of the current network (raw policy, moves sampled at
  temperature 1, no search) against the practice opponents and returns per-opponent summaries plus every game's
  record. A *positions pass* returns, for up to 256 positions, the top policy moves, probabilities over legal
  moves, the **value-head output**, and on request every layer's activations. The session continues until every
  pass is used.
- **Practice opponents** (never used to score anything): a uniform random mover, and a 1-ply "practice bot"
  (plays mate if available, else the capture with the best victim-minus-attacker value, else a check, else a random
  move). The evaluation anchors (§8.2) are never shown to the model.
- **Audit:** every turn's event stream (no tool use) and every Python run (no network execution, no
  autograd / optimizer / fitting, no search) are checked, automatically and by reading. A violating attempt is
  excluded, reported and replaced by a new attempt. Attempts are never replaced for being weak.
- **Cost logging:** tokens and Claude Code's API-equivalent cost per turn, logged separately from GPU time.

**[D5] Attempts and effort.** **Decided (Justin, 2026-11-08):** **3 attempts at budget 0 and 5 at budget 50**
(the checklist's "0 and maximum"; 5 at budget 50 so that each of the 5 WSP seeds has its own network, per D3 and
D7), at effort `xhigh`, the level used in the WSP Go experiment, so the results are comparable. Together with D4's
five distillation sessions, the LLM budget must cover ten 50-pass-sized sessions and three 0-pass sessions; they are
spread over as many weeks of the subscription allowance as needed.

**Evaluation of every written network** (Phase 5 harness): raw policy (top-1 agreement with Stockfish), raw value
(accuracy and calibration), and the full agent with search (§8).

**[D6] GO / NO-GO criterion ("clearly better than random initialization").** **Decided (Justin, 2026-11-08):**
**GO** if at least
3 of the 5 budget-50 networks, as full agents (200 simulations, no noise, most-visited move), score at least 60%
against the random-initialization agent over 200 games (100 openings × both colors), with the 95% confidence
interval's lower bound above 50%. Results are reported to Justin either way, and Phase 7 starts only with his
approval.

## 7. Game budget and seeds

- **[D7] Seeds and compute.** **Decided (Justin, 2026-11-08):** **5 seeds per arm** (15 runs).
  - **Where:** all runs on Modal, every run on one container type (CPU core count and a low-cost GPU such as an L4,
    chosen in Phase 3 for cost per game), so hardware is identical across arms.
  - **Game budget per run** *(fixed after Phase 3)*: the largest round number of games such that the whole main
    experiment (the D1 pilot, the D8 calibration run, distillation imitation, 15 runs and their evaluations) is
    estimated to cost at most $192 on Modal (the $240 cap minus 20% slack), at the cost per game measured on
    Modal in Phase 3. (Cap raised from $150 to $240 by Justin, 2026-11-08, before approval.)
  - **Fallback:** if that budget is below 15,000 games per run, Modal runs seeds 1–3 of every arm and Justin's PC
    runs seeds 4–5 of every arm, and the budget is recomputed for 9 Modal runs. The split is always by seed,
    never by arm.
- Seeds: run seeds 1–5 for every arm; pilot seeds start at 100.

## 8. Evaluation

### 8.1 Schedule
Checkpoints are saved at games 0, 250, 500, 1,000 and 2,000 (dense early points, where any head start would show),
and every K self-play games, and are evaluated offline after training (evaluation never slows or influences a
run). K *(fixed after Phase 3)* = budget / 20.

### 8.2 Opponents (fixed anchors)
- Uniform random mover.
- Greedy material-grabber: captures the most valuable piece available, else a random move.
- Stockfish (one pinned version) at fixed node counts 1, 3, 10, 30, 100, 300, 1,000, 3,000 and 10,000
  (machine-independent), and strength-limited at its minimum `UCI_Elo`.
- The anchor ladder is calibrated (anchors play each other) in Phase 5, before any training run.

### 8.3 Games
- Our agent: 200 simulations, no Dirichlet noise, the most-visited move from ply 1. A secondary agent plays the
  raw policy's top move (no search).
- Openings: a fixed suite of 50 positions for checkpoint evaluation and a separate 100 for the GO / NO-GO test.
  Generated once by random 8-ply sequences kept only if Stockfish (depth 20) evaluates them within ±50 centipawns;
  committed to the repo. Each opening is played with both colors: 100 games per anchor per checkpoint.

### 8.4 Metrics
- **Elo** for every checkpoint and anchor from all games jointly (Ordo), with the random mover fixed at 0;
  95% confidence intervals.
- **[D8] Strength threshold (headline metric).** **Decided (Justin, 2026-11-08):** the primary threshold is a
  **Stockfish rung chosen by a calibration run of the random arm**:
  - *Calibration run:* the winning D1 pilot setting, continued to the full game budget on a pilot seed (excluded
    from all analyses), evaluated on the same schedule.
  - *Primary threshold:* a score of at least 50% (full agent) against the strongest Stockfish rung (§8.2) that the
    calibration run reaches, by the rule below, within 60% of its budget. If it reaches no Stockfish rung, the
    primary threshold is 50% against the greedy material-grabber.
  - The threshold is fixed and recorded in [decisions.md](decisions.md) before any WSP or distillation training
    run. Only random-arm data chooses it, so it cannot favor WSP.
  - *Time to threshold* = self-play games at the first of two consecutive evaluation points at or above the
    threshold (two, so one lucky evaluation does not count). Runs that never reach it are censored at the budget.
  - Secondary thresholds: 50% against the greedy material-grabber and against every other Stockfish rung, same
    rule.
- **Policy:** top-1 agreement with Stockfish (depth 20) on a fixed suite of 1,000 middlegame positions.
- **Value:** accuracy and calibration (expected calibration error) on a fixed suite of 1,000 positions with 3–5
  pieces labeled by Syzygy tablebases (win / draw / loss), and correlation with Stockfish's evaluation on the
  middlegame suite.

## 9. Analysis plan

### 9.1 Reported for every arm
Learning curves (Elo vs self-play games, mean over seeds with 95% intervals); initial gap (Elo at game 0); time to
threshold; washout (the first evaluation point after which the WSP − random Elo difference's interval includes 0
at every later point); asymptote (Elo at the final checkpoint).

### 9.2 [D9] What counts as a result
**Decided (Justin, 2026-11-08):** an exact one-sided Mann–Whitney test on time to threshold, 5 runs per arm.
- **Statistic:** U = the number of (WSP run, random run) pairs, out of 25, in which the WSP run is slower. A tie
  (same evaluation point) counts ½. A censored run counts as slower than every run that reached the threshold;
  two censored runs tie.
- **p-value:** exact, over all 252 assignments of the 10 runs' times to the two arms (handles ties).
- **H1 positive:** p < 0.05 in favor of WSP (without ties, U ≤ 4, p ≈ 0.048).
- **H1 negative:** p < 0.05 in favor of random (the same test in the other direction).
- **Inconclusive:** neither.
- **Always reported:** the difference in median time to threshold (WSP − random) with a bootstrap 95% interval.
- **H2** (secondary): the same test, WSP vs distillation.
- **Further analyses** chosen after seeing the data are welcome but are reported as exploratory; they never
  replace the rule above.

### 9.3 Feature survival (WSP arm)
At every checkpoint: each layer's relative change from its written weights, ‖W_t − W_0‖ / ‖W_0‖, in Frobenius
and spectral norm; and, for each channel the model's channel map says computes a feature, the correlation between
that channel's activation and the feature on a fixed probe set of positions.

## 10. Amendments to the checklist (made before any results)

- **Phase 4 gate:** a uniform-policy dummy network with value 0 cannot see that hanging its queen is bad. The gate
  uses a dummy network with a uniform policy and a **material-count value**, and must find mate-in-1 and avoid
  hanging its queen at 200 simulations or fewer.
- **Phase 5 gate:** "two identical checkpoints rate as equal" is tested using the opening suite, since otherwise
  identical deterministic agents replay identical games.
- **Pre-experiment parameters fixed by rule after Phase 3:** game budget, replay window, K, and LR drop steps
  (from §4, §7, §8.1); and, after the D1 pilot and calibration run, the base LR, warmup and primary threshold
  (§4, §8.4). All are recorded in [decisions.md](decisions.md) before any WSP or distillation training run.
- **Value head (amended 2026-11-08, Justin, before any results; superseded the same day by restoring batch
  norm, below).** AlphaZero's value head is 1×1 conv → batch norm
  → ReLU → … With batch norm removed (in every arm), that single ReLU channel died: in a realistic training check
  (16,384 positions, soft move targets, a material-based result target, batch 1,024, LR 0.003) the value head of
  all three seeds died within 300 steps and learned nothing, while the move head learned normally. With the ReLU
  removed, all three seeds learned (held-out value error 0.52 → 0.20–0.25). The 1×1 value conv is therefore
  linear; the parameter count is unchanged (570,379) and nothing else changes. Details: Phase 3 gate report.
- **Value-head health (added 2026-11-08, Justin).** Without batch norm the value head can still fail by saturating
  (tanh pinned at ±1), as seen when memorizing random labels on very small batches. Every training run logs
  `ws_zero.net.value_head_health` (value layer activity, saturated fraction, spread of values), and the
  **Phase 7 gate** also requires a healthy value head throughout the smoke run. If it fails there, the issue is
  revisited with Justin before any experiment.

- **Batch norm restored (amended 2026-11-08, Justin, before any results).** Without batch norm, the amended value
  head still failed: over 3,000 steps of the realistic check one seed in three saturated for good (value pinned at
  ±1 after a rare explosive training step) and a second saturated briefly; the overfit check failed on every seed
  the same way. A cap on the gradient norm prevented it but slowed learning. Justin chose to restore batch norm, as
  in AlphaZero, in every arm, with AlphaZero's original value head (the ReLU after the value conv returns, since
  batch norm removes the reason it was dropped). This departs from the checklist's "batch norm removed or folded to
  identity". How the WSP arm's written weights meet batch norm (the network Claude writes is used in training mode,
  where batch statistics replace the stored ones) is decided before Phase 6. Details: Phase 3 gate report.

- **Pilot rule for a broken value head (added 2026-11-08, Justin, before any results).** With batch norm, the value
  map's single ReLU channel still shut off at the two largest D1 learning rates (0.05, 0.2) in a realistic check
  without warmup (still off after 1,000 steps in 2 of 4 runs); with the protocol's warmup, neither of 2 runs ended
  broken (one shut off and recovered). D1 therefore treats a pilot setting whose value head is broken at the end of
  its pilot run as diverged. "At the end" rather than "ever", because with batch norm a shut-off can recover. The
  network and the learning-rate grid are unchanged. Details: Phase 3 gate report.

## 11. Deviations

None yet. Any change after approval is recorded in [decisions.md](decisions.md) with its date, reason and
Justin's approval.
