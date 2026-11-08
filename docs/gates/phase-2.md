# Phase 2 gate report: chess environment

**Date:** 2026-11-08. **Status: PASSED.** Justin accepted pgx's repetition convention (open issue 1) the same day.

## What was built

- `ws_zero.game.env`: batched chess on **pgx 2.6.0** (JAX): `init`, `step` (finished games are frozen), `observe`
  (B × 8 × 8 × 119), `legal_action_mask` (B × 4,672), `is_terminal`, `terminal_value`, plus FEN conversion.
  Uses pgx's low-level `Game` so that colors are explicit.
- `ws_zero.game.moves`: mapping between pgx's 4,672 actions and python-chess moves (needed for Stockfish,
  tablebases and the distillation heuristic).
- `ws_zero.game.reference`: an independent reference on python-chess: legal actions, the 119-plane observation
  written from the AlphaZero paper with pgx's conventions, and outcomes.
- [`docs/encoding.md`](../encoding.md): every input plane and every action plane, exactly. The WSP prompt is built
  from it.
- `scripts/gate_phase2.py`: the gate check.

## Tests

65 unit tests pass (`uv run pytest`), including exact action numbers for castling, en passant and every
promotion type for both colors, and agreement with pgx's legal-action mask on targeted positions.

## Gate results

2,000 uniformly random games (seed 0), every position checked, in two modes
(raw results: [`phase-2-gate-pgx.json`](phase-2-gate-pgx.json), [`phase-2-gate-fide.json`](phase-2-gate-fide.json)):

| Check | Reference with pgx's repetition convention | Reference with FIDE repetition |
|---|---|---|
| Positions checked | 680,319 | 680,319 |
| Legal-action sets differ | 0 | 0 |
| Move → action → move failures (every legal move) | 0 | 0 |
| Action → move → action failures (every legal action) | 0 | 0 |
| Outcome (terminal, value) differs | 0 | 0 |
| Observation differs | **0** | 48 (0.007%), repetition planes only |

Coverage (occurrences as a legal move, all round-tripped): castling king-side 2,046 (White) / 1,961 (Black),
queen-side 943 / 882; en passant 500 / 488; promotions to queen, rook, bishop and knight 13,501 each (White) /
13,676 each (Black). Games ended by: insufficient material 1,036, fifty-move rule 390, checkmate 298, stalemate 135,
ply cap 84, threefold repetition 57.

The original checklist gate (round trips on every legal move, including castling, en passant and all promotion
types) passes outright.

## Throughput (preview; Phase 3 measures properly)

pgx random play on the Mac's CPU (M4 Pro): about 22,000 positions per second at batch sizes 256, 1,024 and 4,096.
GPU throughput is to be measured on the PC or Modal in Phase 3.

## Issues found and fixed

1. **pgx under JAX ≥ 0.5: repetitions of the start position were never detected.** pgx builds its Zobrist hash
   tables from JAX's random generator at import, but hard-codes the start position's hash from an older generator
   (JAX 0.5 changed the default, `jax_threefry_partitionable`). Confirmed: the hard-coded hash matches only with
   the old setting. Fixed in `env.initial_state`, which stores the correctly computed hash; regression test
   `test_repetition_of_the_start_position_is_detected`.
2. **Scalar planes differ in the last bit.** XLA computes planes 113 and 118 (÷512, ÷100) with possible rounding
   in the last bit. The gate compares those two planes within 1e-6 and every other plane exactly.

## Open issues

1. **Repetition identity: resolved, pgx's convention accepted (Justin, 2026-11-08).** pgx treats a position right
   after a two-square pawn move as different from the same position later, even when no en passant capture is
   possible; FIDE treats them as the same. Measured effect: 0.007% of positions have different repetition planes,
   and no game outcome differed in 2,000 games (in principle a threefold draw could be delayed by one repetition).
   Documented in [encoding.md](../encoding.md).
2. pgx's low-level API (`pgx._src`, `pgx.experimental`) is used; safe because pgx is pinned to 2.6.0.
3. PyTorch is still installed (seeding and run logging import it); it moves out in Phase 3 when the network is
   written in JAX.
