# DALI_modulation — Ablation Study Report

**Date:** 2026-03-17
**Status:** Training complete; gauntlet evaluation in progress
**Objective:** Identify the contribution of each design component in the frozen-base + modulation-head architecture through rigorous ablation.

---

## 1. Research Question

The `value_based` agent (TD(0), 15→64→64→1) is a strong Leduc Hold'em policy. Can we improve it by adding a trainable modulation head conditioned on opponent statistics?

If yes, *which components drive the improvement*?

- **Opponent statistics** — does the agent actually exploit the 7-dim stats, or is the gain just from more compute?
- **Frozen base** — does pre-training stability help, or does end-to-end finetuning do better?
- **State-conditioned gate** — can the agent learn *when* to use opponent information?

---

## 2. Study Agents

All modulation agents share the same frozen base: `agents/value_based/checkpoint.pt`.
Training: 300K episodes, lr=1e-4, batch=32, Adam, weighted opponent sampling (CFR/heuristic 3× oversampled).

| ID | Architecture | Seeds | Purpose |
|----|-------------|-------|---------|
| `value_based` | TD(0), 15→64→64→1, no modulation | — | Baseline |
| `full_modulation` | Frozen base + Δ(s, stats_7dim) [22→32→32→1] | 0, 1, 2 | Primary proposal |
| `state_only` | Frozen base + Δ(s) [15→32→32→1], no stats | 0, 1, 2 | Ablation A: removes opponent stats |
| `finetuned_base` | Unfrozen base + Δ(s, stats_7dim) | 0, 1, 2 | Ablation B: removes frozen base constraint |
| `gated_modulation` | Frozen base + g(s,stats)×Δ(s,stats) | 0 | Gate variant: state-selective modulation |

**Total study agent instances:** 11
**Checkpoint paths:** `DALI_modulation/agents/{agent}/outputs/seed_{N}/checkpoint_final.pt`

### Design Decisions

- **Frozen base**: Provides stable value estimate as residual target. The base was trained on a balanced pool; we only need to learn the *correction* for each opponent style.
- **7-dim stats**: Beta-Bernoulli smoothed: preflop_fold_rate, preflop_raise_rate, flop_raise_rate, preflop_fold_to_raise, flop_fold_to_raise, raise_after_raise_rate, confidence (effective sample size proxy).
- **22-dim mod input**: [15-dim game state ‖ 7-dim stats] — gate and mod head both see state+stats, enabling state-selective modulation.
- **Near-zero output init**: Ensures all modulation heads start close to zero, so early training approximates the frozen base.
- **Weighted sampling**: CFR (25%) + heuristic (25%) + 6 rule-based (8.3% each). Oversamples difficult opponents to reduce cold-start failure modes observed in v1.

---

## 3. Pool Agents (Fixed Gauntlet)

The **fixed gauntlet** (vs round-robin) ensures each study agent's score is independent of what other study agents are in the evaluation. Scores are directly comparable across seeds and training runs.

| ID | Type | Notes |
|----|------|-------|
| `cfr` | Tabular CFR+, Nash equilibrium | Strongest strategic opponent |
| `heuristic` | Rule-based heuristic | Structured, beatable |
| `tight_passive` | Rule-based | Fold-heavy |
| `tight_aggressive` | Rule-based | Raise-heavy, fold averse |
| `loose_passive` | Rule-based | Call-heavy |
| `loose_aggressive` | Rule-based | Manic raiser |
| `maniac` | Rule-based | Always raise |
| `random` | Rule-based | Uniform random |
| `adaptive_value` | NN, 19-dim (15+4 stats), trained | Stats-aware baseline; gives cold-start context |
| `opp_encoder_v1` | NN, frozen base + learned opponent encoder | Richer opponent representation baseline |
| `reinforce` | Policy gradient (REINFORCE), 200K eps | New baseline trained for this study |
| `actor_critic` | A2C, shared trunk, 200K eps | New baseline trained for this study |
| `dqn` | DQN, target net + replay, 200K eps | New baseline trained for this study |

**Total pool agents:** 13
**Total matchups:** 11 study instances × 13 pool agents = **143 matchups**

---

## 4. Evaluation Protocol

### Per Matchup
- **10,000 hands**, position-alternated (seat 0 ↔ seat 1 every hand)
- **Session length:** 100 hands; stats tracker reset at session boundary
- **Stats tracker:** Beta-Bernoulli smoothed (`OpponentStatsTracker` from `opp_stats_input_augmentation_v1`)
- Prior strength: 20.0 (soft start; confidence ≈ 0 at hand 0, ≈ 0.83 at hand 100)

### Per-Hand JSONL
Each matchup produces `results/{agent}_seed{N}/vs_{pool}.jsonl` with one JSON object per hand:
```json
{"hand_id": 0, "reward": 1.0, "cumulative": 1.0, "position": 0,
 "session": 0, "hand_in_session": 5, "confidence": 0.34}
```

### Summary JSON
`results/{agent}_seed{N}/vs_{pool}.json`:
```json
{
  "study_agent": "full_modulation", "seed": 0, "pool_agent": "cfr",
  "chips_per_round": 0.042,
  "ci_95_low": 0.010, "ci_95_high": 0.075,
  "cold_mean": 0.023, "warm_mean": 0.051,
  "n_hands": 10000
}
```

### Aggregate Metrics (per study agent, across 13 pool opponents)
| Metric | Formula |
|--------|---------|
| `avg` | Mean chips/round across 13 opponents |
| `std` | Std dev of chips/round across 13 opponents |
| `robustness` | `avg − 1.5 × std` — penalizes fragility |
| `worst_case` | Min chips/round across opponents |
| `cold_mean` | Mean chips/round, hands 0–19 (stats cold) |
| `warm_mean` | Mean chips/round, hands 20–99 (stats accumulating) |

**Primary metric:** `robustness`. Secondary: `avg`, `worst_case`.

### Cold-Start Decomposition
Hands 0–19 within each 100-hand session are "cold" (confidence ≈ 0, stats ≈ prior).
Hands 20–99 are "warm" (confidence growing, stats increasingly informative).
Comparing cold_mean vs warm_mean isolates whether improvement requires observing the opponent.

---

## 5. Planned Figures

| Figure | X-axis | Y-axis | Purpose |
|--------|--------|--------|---------|
| **Fig 1: Per-opponent bar chart** | Pool opponent | chips/round ± CI | Compare all 5 study agents across opponents |
| **Fig 2: Bankroll trajectory** | Hand number (0–10K) | Cumulative chips (start=0) | Visual: does full_modulation pull away from state_only? |
| **Fig 3: Cold vs warm decomposition** | Study agent | chips/round | Bar chart: cold (grey) vs warm (blue) |
| **Fig 4: Seed variance** | Study agent | chips/round ± seed std | Show stability of 3-seed estimates |
| **Fig 5: Robustness ranking** | Study agent | robustness score | Primary summary figure |
| **Fig 6: CFR convergence** | Training episode | chips vs CFR | Training curve for all seeds |

---

## 6. Hypotheses and Expected Outcomes

| Comparison | Hypothesis | Interpretation if TRUE |
|------------|-----------|----------------------|
| `full_modulation` > `value_based` (avg) | Stats + frozen base improve policy | Opponent modeling adds value |
| `full_modulation` > `state_only` (avg) | 7-dim stats add signal beyond state encoding | Opponent stats are informative, not just extra capacity |
| `full_modulation` > `finetuned_base` (robustness) | Frozen base provides stable inductive bias | Pre-training constraint is beneficial |
| `gated_modulation` > `full_modulation` (robustness) | State-conditioned gate reduces fragility | Gate suppresses modulation in invariant states |
| `full_modulation`: warm_mean > cold_mean | Improvement requires stats accumulation | Signal is truly opponent-specific, not capacity |

---

## 7. Training Summary

| Agent | Seeds | Episodes | Time (s/seed) | Notes |
|-------|-------|----------|--------------|-------|
| `full_modulation` | 0,1,2 | 300K | ~322 | delta_mean grows 0.007→0.35 |
| `state_only` | 0,1,2 | 300K | ~298 | Fastest (15-dim vs 22-dim input) |
| `finetuned_base` | 0,1,2 | 300K | ~383 | base_drift ≈ 0.50 at ep300K |
| `gated_modulation` | 0 | 300K | ~402 | High loss spikes vs maniac/loose_agg |
| `reinforce` (pool) | — | 200K | — | REINFORCE MC PG |
| `actor_critic` (pool) | — | 200K | — | A2C, online TD(0) |
| `dqn` (pool) | — | 200K | — | DQN, target net + replay |

---

## 8. Files

```
DALI_modulation/
├── config.json                        Training configuration
├── STUDY_REPORT.md                    This document
├── agents/
│   ├── full_modulation/{agent,train}.py
│   ├── gated_modulation/{agent,train}.py
│   └── ablations/
│       ├── state_only/{agent,train}.py
│       └── finetuned_base/{agent,train}.py
├── pool_agents/
│   ├── reinforce/{agent,train}.py
│   ├── actor_critic/{agent,train}.py
│   └── dqn/{agent,train}.py
├── evaluation/
│   ├── pool.py                        Loads all 13 pool agents
│   ├── gauntlet.py                    Runs all 143 matchups
│   └── results/                       [gitignored] per-matchup JSONL + JSON
└── shared/
```

---

## 9. How to Run

```bash
# Run all 143 matchups (sequential):
python -m DALI_modulation.evaluation.gauntlet --all --rounds 10000

# Run with 8 parallel workers:
python -m DALI_modulation.evaluation.gauntlet --all --rounds 10000 --workers 8

# Single matchup:
python -m DALI_modulation.evaluation.gauntlet --study full_modulation --seed 0 --pool cfr

# Print aggregate report after evaluation:
python -m DALI_modulation.evaluation.gauntlet --report
```
