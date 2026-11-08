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
