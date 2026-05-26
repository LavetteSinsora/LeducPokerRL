# alphazero_v2 — Fixed Opponent Training + Reduced Architecture

## Hypothesis

The v1 AlphaZero agent degraded progressively (avg −0.919 → −0.983 over 70K–120K episodes)
due to two compounding failures identified in `alphazero_diagnosis_v1`:

1. **Inverted belief updates** (D7): BeliefNet learned "raise = weak hand" from self-play,
   because K was raised most at 70K ep. BeliefNet then inverted this signal as strategy drifted.

2. **Strategy cycling** (D5): Both players share weights in self-play. K−J raise spread
   oscillated: −0.03 → +0.41 → −0.03 → −0.08 without convergence. Non-stationary target.

3. **Portfolio collapse** (D6): b_opp[J/Q/K] diversity ~0.07–0.10 throughout; PIMC search
   used nearly identical opponent models for all imagined hands.

**Fix:** Replace P1 self-play with a frozen opponent pool [value_based, cfr, heuristic],
randomly sampled each episode. Only P0's decisions generate training signal.
This makes Q* targets stationary and gives BeliefNet stable action distributions to learn from.

**Architecture reduction:** d_model 8→4, reduce parameter count to allow faster convergence.
Identify minimal sufficient capacity for Leduc Hold'em.

## Changed Axes

| Axis | v1 (baseline) | v2 (this experiment) |
|------|--------------|----------------------|
| P1 policy | PIMC self-play (shared Q_θ) | Random opponent from {value_based, cfr, heuristic} |
| d_model | 8 | **4** |
| state_hidden | (16, 16) | **(8, 16)** |
| belief_hidden | (32, 32) | **(16, 16)** |
| q_hidden | (64, 64, 64) | **(64, 64)** |

## Success Criteria

- Tournament avg > −0.8 at best checkpoint (vs v1 best of −0.894)
- Robustness (avg − 1.5×std) > −1.1 (vs v1 best of −1.131)
- K−J raise spread monotonically increasing over training (no oscillation)
- b_K rises after P1 raises in D7-style belief trace

## Evaluation Protocol

Standard TournamentCheckpointer: every 10K episodes, 2000 rounds/matchup vs
[heuristic, value_based, adaptive_value, modulated_value, cfr].
Reports checkpoint_best_avg.pt and checkpoint_best_robust.pt.
