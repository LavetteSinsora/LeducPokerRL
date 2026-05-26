# Experiment Standards

> **Authority**: `CLAUDE.md` (root) is the authoritative agent instruction file.
> This document is reference-only — detailed appendix for experiments.
> Training parameters live in `experiments/EVAL_CONFIG.json`.
> In any conflict, `CLAUDE.md` wins.

---

## 1. Training Budget

Standard training configuration for architecture comparison is defined in
`EVAL_CONFIG.json` (episodes: 200,000, lr: 1e-4, batch_size: 32, optimizer: Adam).

### Convergence Check (required before concluding failure)

Before writing "this approach does not work" in a report, verify:

1. **Loss plateau test**: mean loss over last 20% of episodes. If still decreasing
   by more than 5% relative to the previous 20%, training has not converged — extend.
2. **Peak vs final performance**: report both. If peak >> final, it is a training
   stability problem, not a capacity problem.
3. **Multiple seeds** (when claiming failure): run at least 2 seeds.

### Comparison Fairness

Same recipe for both methods: same episodes, same batch size, same optimizer,
same evaluation interval.

---

## 2. Required Output Artifacts

Every experiment run must produce the following in `experiments/<exp_id>/outputs/`.

### 2a. Always Required

| Artifact | Format | Description |
|---|---|---|
| `checkpoint.pt` | PyTorch | Final model weights |
| `train_history.json` | JSON array | One entry per batch: `{episode, loss, type}` |
| `train_config.json` | JSON | All hyperparameters used |
| `training_curve.png` | PNG | Episodes vs smoothed loss |
| `eval_curve.png` | PNG | Episodes vs avg_chips_vs_heuristic |
| `results.json` | JSON | Final numeric metrics (see §3) |

### 2b. Required for Representation Learning Experiments

| Artifact | Format | Description |
|---|---|---|
| `repr_quality_curve.png` | PNG | Reward Spearman ρ and/or hand probe accuracy over training |
| `pca_snapshots/` | PNG dir | PCA scatter at episode 0, mid-training, and final |
| `effective_dim_curve.png` | PNG | Effective dimension (80% PCA variance) over training |
| `linear_probe_results.json` | JSON | Hand accuracy, reward R² from linear probes |

### 2c. Required for Policy / Value Network Experiments

| Artifact | Format | Description |
|---|---|---|
| `tournament_history.json` | JSON | One entry per tournament (every 10K episodes) |
| `checkpoint_best_avg.pt` | PyTorch | Highest tournament_avg_chips across all tournaments |
| `checkpoint_best_robust.pt` | PyTorch | Highest tournament_robustness (primary for promotion) |
| `checkpoint_ep{N}.pt` | PyTorch | Snapshot at each tournament point |
| `entropy_curve.png` | PNG | Policy entropy (bits) per episode |
| `value_error_curve.png` | PNG | TD error or value MSE over training |

### 2d. Recommended

- `eval_vs_baselines.png` — chips vs heuristic, value_based, cfr at final checkpoint
- `confusion_matrix.png` — for classification probes
- `pairwise_distance_scatter.png` — embedding distances vs |ΔR|

### Plotting Standards

- Labeled axes with units, title, legend
- Loss curves: exponential moving average (α=0.95)
- Evaluation curves: raw points + smoothed line
- 150 DPI minimum

---

## 3. Required Metrics in `results.json`

```json
{
  "experiment_id": "string",
  "training_episodes": 0,
  "converged": true,
  "peak_eval_score": 0.0,
  "final_eval_score": 0.0,
  "eval_opponent": "heuristic",
  "eval_rounds": 1000,
  "tournament_avg_chips": null,
  "tournament_robustness": null,
  "loss_final": 0.0,
  "loss_components": {},
  "representation_metrics": {
    "effective_dim_80": null,
    "effective_dim_90": null,
    "reward_spearman_rho_pairwise": null,
    "hand_probe_accuracy": null,
    "hand_probe_chance": 0.333
  },
  "notes": "string"
}
```

Metric definitions: `METRICS_GLOSSARY.md`.

---

## 4. Report Writing Standards

### What Constitutes a Conclusion

A report may claim a method "works" only if:
- Training converged (§1 plateau test passed)
- Final eval score exceeds the relevant baseline
- At least 1000 evaluation rounds used (position-swapped)

A report may claim "does not work" only if:
- Convergence confirmed, budget met, AND peak score also fails to exceed baseline

Otherwise: write "training did not converge" or "insufficient budget" — not "fails."

### Report Sections (required)

1. **Setup**: architecture, loss functions, training recipe
2. **Results table**: numeric metrics vs baselines
3. **Training dynamics**: convergence? oscillation? loss plateau test result
4. **Key finding**: main takeaway in one paragraph
5. **Failure analysis** (if applicable): mechanism of failure
6. **Next steps**: natural follow-up experiment

### Metric Presentation

- State baseline value alongside experiment value
- State chance/random baseline for classification metrics
- Spearman ρ: always specify variant (scalar vs pairwise — see METRICS_GLOSSARY.md)
- 3 decimal places for correlation/accuracy metrics

---

## 5. Forbidden Conclusions

| Banned phrase | Required justification |
|---|---|
| "X does not work" | Convergence confirmed + budget met + peak score also fails |
| "X converged" | Loss change < 5% over last 20% of training |
| "X outperforms Y" | Both converged, same budget, same eval suite |
| "The representation is low-dimensional" | PCA effective dimension reported with sample size |
| "Training is stable" | Loss curve shown; no oscillation > 50% of mean in last 30% |

---

## 6. Sub-Agent Instructions

When implementing and running an experiment:

1. Read `CLAUDE.md` first. Follow its experiment lifecycle and evaluation protocol.
2. Use `TournamentCheckpointer` (see `agents/tournament_eval.py`). Non-negotiable.
3. Use parameters from `experiments/EVAL_CONFIG.json` (200K episodes, etc.).
4. Produce all artifacts in §2a + applicable §2b/§2c.
5. Run convergence check (§1) before writing report conclusion.
6. Populate `results.json` with all keys in §3.
7. If training does not converge within budget, extend by 50% and re-check.
8. Never conclude "fails" without meeting §4 requirements.
9. Run `python experiments/validate_experiment.py experiments/<exp_id>/outputs/` before finalizing.

---

## 7. Tournament Evaluation Protocol

Full specification for the standard evaluation protocol. Parameters in EVAL_CONFIG.json.

### 7.1 Why This Is Required

Single-checkpoint evaluation against heuristic alone is insufficient for architecture
comparison because:
- Final checkpoint ≠ best checkpoint; training dynamics differ across architectures
- Heuristic-only eval cannot distinguish robustness from raw performance
- Without periodic checkpointing, training curves cannot be compared

### 7.2 Evaluation Configuration

| Parameter | Value | Rationale |
|---|---|---|
| Rounds per matchup | 2,000 | SEM ≈ 0.009 chips/round (σ≈0.4); differences ≥ 0.03 meaningful |
| Session length | 100 hands | Matches training distribution for stateful agents (adaptive_value, modulated_value) |
| Tournament interval | 10,000 episodes | ~20 snapshots over 200K budget |
| Opponent pool | All 5 promoted agents | Loaded fresh from `agents/{id}/checkpoint.pt` each tournament |
| Eval mode | `set_train_mode(False)` on all | Greedy action selection |
| RNG isolation | Save/restore torch + Python state | Training stream unaffected |

**Session length note**: `adaptive_value` and `modulated_value` were trained with
PokerSession stats resetting every ~100 hands. Evaluating them in a continuous
2000-hand session is out-of-distribution. `session_length=100` fixes this.

### 7.3 Checkpoint Selection

- **`checkpoint_best_robust.pt`** — primary candidate for promotion.
  Robustness = `avg - 1.5 × std` across per-opponent scores.
  Penalizes fragility: high avg + high variance is worse than moderate avg + low variance.
- **`checkpoint_best_avg.pt`** — secondary. Highest raw average across opponents.
- If `best_avg` and `best_robust` come from very different episodes, investigate
  training stability.

### 7.4 Per-Position Breakdown

`tournament_history.json` matchup entries record `avg_chips_as_p0` and `avg_chips_as_p1`
separately. In Leduc Hold'em, P1 acts second and has a pre-flop information advantage.
Large P0/P1 asymmetry in results indicates positional bias rather than genuine skill.

### 7.5 `tournament_history.json` Schema

```json
[
  {
    "episode": 10000,
    "timestamp": "20260316_143022",
    "elapsed_seconds": 847.3,
    "tournament_avg_chips": 0.312,
    "tournament_robustness": 0.084,
    "tournament_std": 0.152,
    "tournament_worst_case": 0.027,
    "tournament_best_case": 0.521,
    "n_opponents": 5,
    "is_best_avg": true,
    "is_best_robust": false,
    "snapshot_path": "experiments/exp_id/outputs/checkpoint_ep0010000.pt",
    "matchups": {
      "heuristic": {
        "avg_chips": 0.412,
        "avg_chips_as_p0": 0.438,
        "avg_chips_as_p1": 0.386,
        "total_chips": 824.0,
        "rounds_counted": 2000
      }
    }
  }
]
```
