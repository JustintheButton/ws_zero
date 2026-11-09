# Decisions and deviations

Every decision, with its date and who made it. Once any experimental result exists, changes to the protocol are
recorded here as **deviations**, with Justin's approval.

## Decisions

| Date | Decision | By |
|---|---|---|
| 2026-11-07 | Editing LLM: Claude Opus 5.5 (`claude-opus-5-5`). | Justin |
| 2026-11-08 | LLM budget: about half of Justin's weekly Claude subscription usage (not API credits). The WSP harness therefore drives Claude Code in non-interactive mode rather than calling the API with a key. Claude Code reports tokens and API-equivalent cost per call, which is what Phase 9 logs. | Justin |
| 2026-11-07 | Compute: Justin's PC, RTX 5070 Ti (16 GB) and Ryzen 7 (8 cores / 16 threads), Windows. | Justin |
| 2026-11-08 | Workflow: code is written on the Mac and run on the PC; first runs are small-scale on the Mac. | Justin |
| 2026-11-08 | Experiment tracking: Weights & Biases (project `ws_zero`), mirrored from complete local run logs, which stay the source of truth. | Justin |
| 2026-11-08 | Environment: Python 3.12 via uv, exact pins in `pyproject.toml` and `uv.lock`. PyTorch 2.15.0 on both machines; on Linux the CUDA 13.2 build, since the RTX 5070 Ti needs CUDA ≥ 12.8 and the cu128 index stops at torch 2.11. The PC needs a recent NVIDIA driver. | Claude (default; no objection) |
| 2026-11-08 | The PC runs Linux code under WSL2 (Ubuntu), not native Windows. | Claude (recommended; no objection) |
| 2026-11-08 | Protocol D1: base LR and warmup chosen by a pilot on the random arm only (4 LRs × 2 warmups, 20% budget each); gradient clipping only if every pilot setting diverges. | Justin |
| 2026-11-08 | Protocol D2: synchronous chunks (100 games, then train), no gating, batch 1,024, each position sampled about 4 times on average. | Justin |
| 2026-11-08 | Protocol D3: one written network per WSP seed (seed k ← budget-50 attempt k, by run order). Option C (every network × 3 seeds) kept as a follow-up idea. | Justin |
| 2026-11-08 | Protocol D4: distillation = same session as budget-50 WSP but writing a non-searching heuristic; imitation on 1M positions for 2 epochs (not counted as self-play games); one heuristic per seed; imitation fidelity reported. | Justin |
| 2026-11-08 | Protocol D5: 3 WSP attempts at 0 passes and 3 at 50 passes, effort `xhigh`; budgets 10 and 25 skipped (possible follow-up). | Justin |
| 2026-11-08 | Protocol D6: GO if ≥ 2 of 3 budget-50 networks (full agent, 200 sims) score ≥ 60% vs the random-init agent over 200 games, 95% CI above 50%; Justin approves before Phase 7 either way. | Justin |
| 2026-11-08 | Protocol D7: 5 seeds per arm; all runs on Modal (one container type, low-cost GPU); game budget fixed after Phase 3 so the main experiment costs ≤ $120 of the $150 Modal cap; if that gives < 15,000 games per run, the PC runs seeds 4–5 of every arm. Modal academic credits (modal.com/academics) noted as a way to raise the cap. | Justin |
| 2026-11-08 | D7 revised before approval: Modal cap raised from $150 to $240 (plan ≤ $192 with 20% slack). | Justin |
| 2026-11-08 | Protocol D8: primary threshold = 50% vs the strongest Stockfish rung a random-arm calibration run (D1 winner continued to full budget, pilot seed) reaches within 60% of its budget; fallback greedy material-grabber. Finer Stockfish ladder (nodes 1, 3, 10 … 10,000). Extra early evaluations at games 250, 500, 1,000, 2,000. | Justin |
| 2026-11-08 | Protocol D9: exact one-sided Mann–Whitney test on time to threshold (5 vs 5); positive / negative at p < 0.05, else inconclusive; median difference with bootstrap CI always reported; post-hoc analyses allowed but labeled exploratory. | Justin |
| 2026-11-08 | Consistency fix (option A): with 5 seeds (D7) and one network or heuristic per seed (D3, D4), D5 becomes 5 WSP attempts at 50 passes (plus 3 at 0) and D4 five heuristics; D6 becomes GO if ≥ 3 of 5 networks pass. LLM usage: 10 long + 3 short sessions, spread over as many weeks as needed. | Justin |
| 2026-11-08 | Technical fix: LR drops and warmup defined as fractions of the game budget rather than of training steps (steps aren't known in advance under D2); pilot runs follow the full-budget schedule and stop at 20%. | Claude (no objection) |
| 2026-11-08 | **Protocol v1.0 approved** and committed as the pre-registration. | Justin |
| 2026-11-08 | Chess environment: **pgx** (JAX, runs thousands of games in parallel on the GPU; its chess uses AlphaZero's 119-plane observation, 4,672 actions and termination rules) instead of wrapping python-chess. python-chess stays as the reference implementation for the Phase 2 gate and for Stockfish, tablebases and the distillation heuristic. Amends checklist Phase 2; not a protocol deviation (the algorithm is unchanged). Consequence: network and training move to JAX (Flax, Optax), replacing PyTorch. The search implementation (our own JAX MCTS vs DeepMind's mctx) is decided at Phase 4. | Justin |
| 2026-11-08 | Phase 2: fixed a pgx 2.6 bug under JAX ≥ 0.5 (start-position repetitions never detected: hard-coded hash from JAX's old random generator); `env.initial_state` stores the correct hash. | Claude (bug fix, reported) |
| 2026-11-08 | Repetition identity: **accept pgx's convention** (an en passant square counts after every two-square pawn move, even with no capture possible) rather than patching pgx to FIDE. Measured: 0.007% of positions get different repetition planes; no game outcome differed in 2,000 games; identical across arms. Documented in docs/encoding.md. | Justin |
| 2026-11-08 | **Phase 2 gate passed** (docs/gates/phase-2.md). | Justin |
| 2026-11-08 | Fixed a second pgx 2.6 bug: `from_fen` keeps the default move history, so a position set up from FEN showed the standard start position one ply back. `env.state_from_fen` clears the history. Self-play (always from the start position) was unaffected; FEN-based evaluation positions would have been. Regression test added. | Claude (bug fix, approved for commit) |
| 2026-11-08 | Phase 3: the network is written in plain JAX (no Flax): a flat dict of named weight arrays, the same names the WSP prompt will use; optax for the optimizer. PyTorch removed from the project. Modal client added as a development dependency. | Claude (default; reported) |
| 2026-11-08 | Value-head instability (Phase 3): when memorizing random win/draw/loss labels, about 1 seed in 3 loses the value head for good, under every variant tried (LR 0.003 and 0.0125, warmup, no ReLU after the value conv, gradient clipping at 1.0). **Keep the protocol unchanged**; log value-head activity in every run; the Phase 7 smoke run must show the value head staying alive; revisit with real-training evidence if it dies. Recorded as a checklist amendment (protocol §10). | Justin |
| 2026-11-08 | **Value head amended: no ReLU after the 1×1 value conv.** The realistic check (scripts/check_value_head.py) showed the original value head dying in all 3 seeds within 300 steps while moves were learned; with the ReLU removed all 3 learned. Approved by Justin on the condition that seeds 1 and 2 also learn (error halved); condition met (seed 1 at 900 steps). Supersedes the "keep the protocol unchanged" row above. Value-head health logging and the Phase 7 check stay. The Phase 3 gate now uses spread-out move targets and realistic results (Justin's suggestion); the random-label version is kept as an option. | Justin |
| 2026-11-08 | Modal GPU benchmark run (T4, L4, A10, L40S, A100-40GB, H100; docs/gates/phase-3-bench-modal.json). Cost under $2. | Justin |
| 2026-11-08 | Compute raised: 11 Modal workspaces (about $35 of credit each, $330 in all). Hard limit of **$30 per workspace**, never exceeded: set by Justin in each workspace's Modal settings, and enforced again by the run launcher. Credentials stay on Justin's machine (never in the repo or logs). Changing the protocol's $240 cap (D7) is decided when the game budget is fixed. | Justin |
| 2026-11-08 | **Batch norm restored in every arm**, as in AlphaZero, with AlphaZero's original value head (conv → batch norm → ReLU). Overnight runs without batch norm: over 3,000 steps of the realistic check one seed in three saturated for good; a gradient-norm cap of 5 prevented it but slowed move learning. Options offered: an always-on cap (recommended), cap 5 now, batch norm, no change; Justin chose batch norm. Departs from the checklist ("batch norm removed or folded to identity"). Supersedes the value-head amendment above. Implementation choices (conventions, not separate decisions): no bias on convs followed by batch norm; running statistics with momentum 0.1, ε 1e-5; zero-initialized scale of each block's last batch norm instead of a zeroed conv; no weight decay on batch norm scales and offsets; value-map activity added to value-head health. | Justin |
| 2026-11-08 | **Pilot rule: a broken value head counts as diverged.** In the D1 pilot, a learning rate whose value head is broken at the end of its pilot run (value map or value layer active on ≤1% of units, or ≥50% of values saturated) cannot be chosen. Evidence (Phase 3 report): at LR 0.05 and 0.2 without warmup the value map's ReLU was still off after 1,000 steps in 2 of 4 runs; with warmup, 0 of 2; removing the ReLU did not help (the value saturated instead). Options offered: this rule (recommended), no change, drop 0.2 from the grid, remove the ReLU. Network and grid unchanged. | Justin |
| 2026-11-08 | **Phase 4 search: mctx** (DeepMind's JAX search library), pinned to 0.0.71; checked to install and run with JAX 0.12.0.1 (toy search: correct move found, illegal moves never visited). Its defaults match AlphaZero's exploration constants (1.25, 19652) and the protocol's root noise (0.25, α 0.3). Its default treatment of values (untried moves scored as the lowest value at the node, all values rescaled to [0, 1]) differs from AlphaZero and is replaced by AlphaZero's, as protocol §3 requires. Options offered: mctx (recommended) or our own search. Known cost: mctx keeps statistics for all 4,672 moves at every tree node (about 2 GB for 100 games at 200 simulations), which limits packing many runs onto one GPU. | Justin |
| 2026-11-08 | **Untried moves in the search score 0 on the −1…+1 scale ("equal").** The published AlphaZero pseudocode scores them 0 on its 0…1 scale (a loss), which with a uniform policy keeps the search on its first-tried move (the Phase 4 gate's dummy, and the random arm's start, have uniform policies; this would handicap the random arm). "Equal" is the AlphaGo Zero paper's rule. Options offered: equal (recommended), loss (literal), the worst tried value (mctx's default). Illegal moves are excluded below the root by the same piece of the search. | Justin |
| 2026-11-08 | Lab notebook page: `lab/template.html`, built by `scripts/build_lab.py` from pgx positions checked against the reference; one section added per phase. | Justin |

## Open questions for the protocol

Raised while reading the brief and the WSP logs; each needs a decision in `protocol.md` before any experiment.

1. **Phase 4 gate wording.** A uniform dummy network with value 0 finds mate-in-1 (terminal values) but has no
   signal to avoid hanging its queen. The gate needs a dummy network whose value counts material.
2. **Fair initialization without batch norm.** An unnormalized ResNet trained with SGD at LR 0.2 is likely
   unstable. The random arm needs a sound no-batch-norm scheme (e.g. zero-initialized last layer of each residual
   branch, or residual scaling), and a fixed warmup length, so that it is not handicapped.
3. **Evaluation openings.** With no noise and argmax after ply 30, identical checkpoints replay identical games.
   Evaluation needs a fixed opening suite.
4. **When to fix the game budget.** Throughput is measured in Phase 3, after the protocol. Proposal: approve
   everything else now and fix the game budget right after the Phase 3 benchmark, still before any experiment.
6. **Written weights and batch norm (before Phase 6).** In training mode batch norm rescales each channel by the
   batch's statistics, which Claude cannot control, so a written network would compute something different from
   what Claude wrote once training starts. Proposal: Claude writes the network without batch norm (convs with
   biases), and before training the harness sets each batch norm to the identity on sample positions (running
   statistics = the channel's measured mean and variance, scale = its standard deviation, offset = its mean plus the
   written bias), so training starts from exactly the written network.
5. **Undefined terms:** "clearly better" at the GO / NO-GO; "matched effort" for the distillation arm; the
   strength threshold for the headline metric; K.
6. **Practice vs. evaluation opponents.** The WSP model's practice opponents must not be the opponents that score
   it (as in the WSP Go experiment).
7. **Games passes:** do practice games use the raw policy (as in the WSP Go experiment) or the full agent with
   search?
8. **Positions passes** should return value-head outputs as well as policy outputs.
9. **Chess library in the WSP sandbox.** Following the Go experiment (no Go library), the sandbox has no
   python-chess; the model may write its own rule code that neither runs the network nor searches.
10. **WSP sandbox platform.** The WSP sandbox uses bubblewrap, which is Linux-only, so WSP sessions run under WSL2
    (or in Docker on the Mac).
11. **Model difference.** The WSP results were obtained with GPT-6 Astra. Whether Claude Opus 5.5 can write useful
    weights is untested; the protocol and writeup should say so.

## Follow-up ideas

Not part of the pre-registered protocol; candidates once the main experiment is done.

- **Every written network × 3 seeds** (from D3, 2026-11-08). Train each budget-50 WSP network with 3 seeds of its
  own (9 WSP runs instead of 3). This separates two sources of variation: how much the written networks differ from
  each other, and how much training luck matters for any one network. Costs 6 extra full runs.
- **Pass budgets 10 and 25** (from D5, 2026-11-08). As in the WSP Go experiment, would show how a written network's
  strength grows with the number of passes.

## Notes from the WSP logs

Read 2026-11-08 from github.com/RikiyaT/lm_understand-logs (commit `4d09ba6`): the top-level README, `harness/`,
and Experiment 6 (9×9 Go from scratch), which is the template for our Phase 6.

- **Harness.** Each turn is one Codex CLI call (`codex exec`, then `codex exec resume`). All built-in tools are
  off and the system prompt is one line ("Answer the user's request."). The model ends each reply with one fenced
  block, ```` ```python ````, ```` ```games ```` or ```` ```positions ````, which the harness runs and answers with
  the printed output. Each turn's event stream is audited for zero built-in tool use (1,125 / 1,125 turns passed).
- **Sandbox.** bubblewrap, no network, no GPU, a `/work` folder kept between runs; numpy, scipy, scikit-learn,
  PyTorch (CPU), pandas, sympy, networkx; no game library. 900 s per run, output cut at 20,000 characters, up to
  120 runs. A prelude defines tensor `load` / `save` helpers and index helpers.
- **Prompt sections.** TASK, THE GAME, THE NETWORK, HOW IT WILL BE MEASURED, WHAT YOU HAVE, RULES, HOW TO FINISH.
  Forbidden: (a) training or fitting weights to data, (b) search, (c) running the network outside passes,
  (d) internet. Allowed: writing weights from its own design, game-rule code that neither runs the network nor
  searches, and analysis of pass results.
- **Passes.** A games pass plays up to 128 games against practice opponents and returns full records. A positions
  pass returns outputs on up to 256 positions, optionally with hidden activations. The session continues until all
  passes are used.
- **Budgets 0, 10, 25, 50, one run each.** Budget 0 still allowed 120 Python runs; the model used one. The blog's
  "0 forward passes: one Python run" describes what happened, not the rule.
- **Cost.** Budget 50: 90 turns over about 4 hours, about 8.1M input tokens (97% cached) and 92k output tokens,
  roughly $5–10 of Opus 5.5 API-equivalent usage. Budget 0: about 42k input and 32k output tokens.
- **Scoring** used 2,000 games against each hidden opponent, which the model never played.
