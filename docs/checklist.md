# AlphaZero Chess × Weight-Space Programming: implementation checklist

Transcribed from the original brief (PDF). Ticked items link to where the work or decision is recorded.

## Project goal

Implement the original AlphaZero algorithm (Silver et al., 2017/2018) for chess from scratch, faithful to the
paper's algorithm but at reduced scale. Then test whether Weight-Space Programming (WSP; Takehi, Teoh, Zhang &
Khattab, MIT CSAIL, Nov 2026), in which a frontier LLM writes network weights directly without gradient descent,
produces a better initialization for AlphaZero training than random initialization. The core research question is
sample efficiency in early training, not final superhuman strength.

## Rules for the agent

- Work phases in the order given. Do not start a phase until the previous phase's **GATE** passes.
- Items marked **[HUMAN]** require a decision from Justin. Stop and ask; do not decide them yourself.
- The protocol in Phase 1 is pre-registered. Never change it after any results exist without explicit approval,
  and record every deviation.
- Every arm of every experiment uses identical hyperparameters, code and evaluation; only the initialization
  differs.
- Write unit tests for every component listed with a test. Pin all package versions. Seed everything. Log every run.
- At each GATE, write a short report: what was built, test results, throughput numbers, open issues.

## Reference specification

| Component | Original paper | This project |
|---|---|---|
| Input | 119 planes: 8 history × 14 (6 piece types per side + 2 repetition) + 7 constant (colour, move count, 4 castling rights, no-progress count) | Same |
| Policy output | 8×8×73 = 4,672 logits (56 queen-style, 8 knight, 9 underpromotion per square); illegal moves masked before softmax | Same |
| Network | 19 residual blocks, 256 filters, batch norm | ~6 blocks, 64 filters; batch norm removed or folded to identity (all arms). *Amended 2026-11-08 (Justin): batch norm kept, as in AlphaZero; see protocol §10.* |
| Search | 800 sims/move; PUCT with c(s) = log((N+19652+1)/19652) + 1.25 | 800 sims/move, same formula. *Amended 2026-11-08 (Justin): raised from ~200; see protocol §10.* |
| Exploration | Root Dirichlet α = 0.3, ε = 0.25; sample moves for first 30 plies, then argmax | Same |
| Loss | (z − v)² − πᵀ log p + L2 | Same |
| Optimizer | SGD, momentum 0.9, weight decay 1e-4, LR 0.2 → 0.02 → 0.002 → 0.0002 | Same, plus a short low-LR warmup in every arm |
| Training | Continuous updates, no gating, no symmetry augmentation; batch 4,096 | Same rules; batch sized to GPU |
| Game length | Draw adjudicated at 512 plies | Same |

## Phase 1: Setup and pre-registered protocol

- [x] Choose the editing LLM and set the API budget. **[HUMAN]** Claude Opus 5.5 (`claude-opus-5-5`); about half of
  Justin's weekly Claude subscription usage. See [decisions.md](decisions.md).
- [x] Secure GPU compute (lab cluster if available). **[HUMAN]** Justin's PC: RTX 5070 Ti, Ryzen 7 (8 cores),
  Windows with WSL2.
- [x] Read the MIT team's released logs (github.com/RikiyaT/lm_understand-logs) for their harness and prompts.
  Notes in [decisions.md](decisions.md#notes-from-the-wsp-logs).
- [x] Create the repo: config files, experiment tracking (e.g. Weights & Biases), seed control, pinned dependencies.
- [x] Write the protocol document and get it approved before any experiment: **[HUMAN]** Approved 2026-11-08:
  [protocol.md](protocol.md) v1.0.
  - architecture, simulations per move, LR schedule with warmup
  - self-play game budget per run, number of seeds per arm
  - evaluation schedule, metrics and anchor opponents

## Phase 2: Chess environment

*Amended 2026-11-08: the environment is **pgx** (GPU-vectorized, AlphaZero encodings built in) rather than a
python-chess wrapper; python-chess is the reference it is checked against. See [decisions.md](decisions.md).*

- [x] Wrap pgx chess behind a generic, batched Game interface (initial state, legal actions, next state, terminal
  value, observation). *(Originally: wrap `python-chess`.)* `src/ws_zero/game/env.py`
- [x] Check pgx's 119-plane observation against a reference encoder written from the AlphaZero paper on
  python-chess (board flipped when Black is to move), and document every plane exactly (the WSP prompt needs it).
  `src/ws_zero/game/reference.py`, [encoding.md](encoding.md)
- [x] Map pgx's 4,672 actions to and from python-chess moves. Queen promotions use queen-style planes; N/B/R
  promotions use underpromotion planes. `src/ws_zero/game/moves.py`
- [x] Check pgx's terminal handling against python-chess: checkmate, stalemate, insufficient material, 50-move
  rule, threefold repetition, 512-ply cap.

> **GATE:** encode → decode round-trips exactly on every legal move across thousands of random positions,
> including castling, en passant and all promotion types; and pgx's legal moves, observations and game results
> agree with the python-chess reference across thousands of random games.
>
> **PASSED 2026-11-08:** 680,319 positions from 2,000 games, zero mismatches (pgx's repetition convention
> accepted). [Report](gates/phase-2.md).

## Phase 3: Network

- [x] Small ResNet with masked-softmax policy head and tanh value head. `src/ws_zero/net.py`
- [x] Benchmark throughput: network evaluations per second and estimated self-play games per hour. Mac and six
  Modal GPU types. [Report](gates/phase-3.md)

> **GATE:** the network overfits a small batch of positions; throughput numbers are reported so the experiment
> timeline can be checked.
>
> **Passed 2026-11-08** with batch norm restored (Justin's amendment): every seed, both kinds of targets; the value
> head also stays healthy through 3,000 steps of realistic training at LR 0.003. At LR 0.05 and 0.2 it can shut
> off (mostly prevented by warmup); a D1 pilot rule excludes any rate that leaves it broken. [Report](gates/phase-3.md)

## Phase 4: MCTS

- [ ] PUCT search with the logarithmic c(s), root Dirichlet noise and the 30-ply temperature schedule.
- [ ] Unit test that values flip sign at every level during backup (the most common AlphaZero bug).
- [ ] Batch network evaluations across many parallel games.

> **GATE:** with a uniform dummy network, the search finds mate-in-1 and avoids hanging its queen at modest
> simulation counts.

## Phase 5: Evaluation harness

- [ ] Stockfish over UCI, both strength-limited and at fixed node counts.
- [ ] Weaker anchors below Stockfish's minimum strength: random mover and greedy material-grabber.
- [ ] Syzygy 3–5 piece tablebases (~1 GB); skip 6-piece (~150 GB) for now.
- [ ] Fixed test suites: middlegame positions labeled by deep Stockfish (best move, eval); endgame positions
  labeled by tablebases.
- [ ] Metrics: policy top-1 agreement with Stockfish; value accuracy and calibration; Elo with confidence intervals
  (Ordo or BayesElo) at fixed simulation counts.

> **GATE:** two identical checkpoints rate as equal within error bars.

## Phase 6: Write-only WSP pilot

- [ ] Build the WSP harness: Python sandbox with tensor read/write, a counted forward-pass budget, full logging of
  every message.
- [ ] Spec prompt: rules, architecture, input/output encoding. The editing LLM is **not allowed**:
  - training or fitting weights to data
  - any search inside the network
  - engine access or internet access
- [ ] Run several attempts at 0 forward passes and at the maximum budget.
- [ ] Evaluate every written network with the Phase 5 harness: raw policy, raw value, and full agent with search.

> **GO / NO-GO [HUMAN]:** Is the written network clearly better than random initialization? Report results and
> stop. Do not continue to Phase 7 without approval.

## Phase 7: Training loop

- [ ] Replay buffer (window of recent games), loss, checkpointing, continuous training without gating.

> **GATE:** a short random-init smoke run shows falling loss and beats the random mover. *Amended 2026-11-08:*
> and the value head stays alive throughout (see the Phase 3 report).

## Phase 8: Controls

- [ ] **Distillation arm:** the LLM writes its chess strategy as a Python heuristic; the network is pretrained to
  imitate it with matched effort.
- [ ] **Random arm:** standard initialization.
- [ ] **WSP arm:** the network written in Phase 6.

## Phase 9: Main experiment

- [ ] 3 arms × at least 3 seeds, identical hyperparameters.
- [ ] Evaluate every K self-play games against the fixed anchors.
- [ ] Log LLM API cost separately from GPU compute.

## Phase 10: Analysis and writeup

- [ ] Learning curves with confidence intervals. Report:
  - initial gap (strength at step 0)
  - time to threshold: self-play games to reach a fixed strength (headline metric)
  - washout: whether and when curves converge
  - asymptote
- [ ] Track whether the LLM-written features survive training (how much the hand-built units change, and whether
  they keep their function).
- [ ] Release code, prompts and logs.
