# Pre-registered protocol

**Status: not started.** To be drafted together and approved by Justin before any experiment. After approval,
this file changes only through a dated deviation recorded in [decisions.md](decisions.md).

Sections to fill in (from the checklist and the open questions in [decisions.md](decisions.md#open-questions-for-the-protocol)):

1. Research question and hypotheses
2. Architecture, including the no-batch-norm initialization for the random arm
3. Search: simulations per move and exploration settings
4. Training: LR schedule with warmup, batch size, replay window
5. Arms: random, WSP, distillation (including "matched effort")
6. WSP pilot: harness, prompt, budgets, number of attempts, the GO / NO-GO criterion
7. Self-play game budget per run, and number of seeds per arm
8. Evaluation: schedule (K), anchor opponents, opening suite, metrics, strength threshold
9. Analysis plan: initial gap, time to threshold, washout, asymptote, feature survival
10. What would count as a positive, negative or inconclusive result
