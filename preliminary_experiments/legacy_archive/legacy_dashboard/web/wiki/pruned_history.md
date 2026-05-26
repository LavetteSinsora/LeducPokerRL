# Pruned History Agent

> Pruned action history features with opponent stats -- underperformed due to insufficient training budget and undersized network.

| Property | Value |
|----------|-------|
| **ID** | `pruned_history` |
| **Parent** | adaptive_value |
| **Round** | 3 |
| **Rank** | #15 / 17 |
| **Avg Score** | -0.691 |
| **Robustness** | -1.781 |

---

## Motivation

Round 2's adaptive_history agent (35-dim input, 128 hidden units) combined opponent stats with action history features but underperformed its parent adaptive_value. Analysis suggested two problems: (1) the 16 action history features included permanently-zero fold counts (fold ends the hand immediately, so fold counts in mid-hand action history are always zero), and (2) the 128-hidden-unit network may have been over-parameterized for the available training data.

The pruned_history agent tests a more efficient approach:
- **Prune useless features**: Remove the 4 always-zero fold count features (2 per round x 2 rounds), reducing history features from 16 to 12
- **Use parent's network size**: Keep 64 hidden units (same as adaptive_value) instead of scaling up to 128

The hypothesis: a leaner observation space with matched network capacity would learn more efficiently than the bloated adaptive_history.

---

## Architecture

```
ValueNetwork(31 -> 64 -> 64 -> 1)
  - Input: 15-dim base + 4-dim opponent stats + 12-dim pruned history = 31 dimensions
  - Architecture: 2-layer MLP with ReLU, 64 hidden units
  - Parameters: ~6,300
```

### Observation Space Breakdown (31 dimensions)

**Base features (15 dims)**: Same as ValueBasedAgent
- Hand encoding (3): one-hot J/Q/K
- Board encoding (4): one-hot J/Q/K + no-board flag
- Pot sizes (2): normalized by max chips (13)
- Turn/position/round/terminal/pair/raises (6): game state indicators

**Opponent stats (4 dims)**: Same as AdaptiveValueAgent
- fold_rate, raise_rate, fold_to_raise_ratio, confidence

**Pruned action history (12 dims)**: 6 features per round x 2 rounds
- `player_call_count / total_actions`
- `player_raise_count / total_actions`
- `opponent_call_count / total_actions`
- `opponent_raise_count / total_actions`
- `total_actions / MAX_ACTIONS_PER_ROUND` (normalized by 6)
- `has_raise_flag` (binary)

**Pruned features** (removed from adaptive_history's encoding):
- `player_fold_count` (always 0 -- fold ends the hand)
- `opponent_fold_count` (always 0 -- fold ends the hand)

### Comparison with Related Agents

| Agent | Obs Dims | Hidden | History Features | Stats |
|-------|----------|--------|-----------------|-------|
| adaptive_value | 19 | 64 | None | Yes |
| adaptive_history | 35 | 128 | 16 (with folds) | Yes |
| **pruned_history** | **31** | **64** | **12 (no folds)** | **Yes** |

---

## Training Configuration

| Parameter | Value |
|-----------|-------|
| Training Method | Session-based self-play (via PrunedHistoryTrainer) |
| Episodes | ~667 sessions x 30 hands = ~20,000 hands |
| Learning Rate | 1e-4 (Adam) |
| Batch Size | 30 hands/session |
| TD Method | TD(0) with stats + history carry-forward |
| Training Time | 87.8 seconds |
| Updates | 333 |
| Final Loss | 14.56 |
| Eval vs Heuristic | +0.07 |

**Critical issue**: The agent received only 667 sessions -- the default budget. The planned budget was 2,000 sessions, but this was not configured correctly for the tournament training run. This budget shortfall is the primary cause of poor performance.

---

## Tournament Results

| Opponent | Score |
|----------|-------|
| heuristic | -1.454 |
| value_based | -1.560 |
| adaptive_value | -1.745 |
| aux_value | -0.209 |
| actor_critic | +0.151 |
| history_value | +0.004 |
| decay_adaptive | +0.632 |
| nstep_value | -1.064 |
| entropy_ac | -1.256 |
| pop_adaptive | -0.233 |
| adaptive_history | -0.879 |
| target_value | -0.645 |
| td_variant | -0.745 |
| modulated_value | -1.050 |
| curriculum | +0.334 |
| extended_adaptive | -1.335 |

Only beats 4 of 16 opponents (actor_critic, history_value, decay_adaptive, curriculum). Loses to all strong agents by large margins.

---

## Diagnosis & Findings

### 1. Budget Impact -- 2000 Sessions vs 667

The most impactful finding: the training budget was drastically insufficient.

| Config | Sessions | Avg Score |
|--------|----------|-----------|
| Original (tournament) | 667 | -1.319 |
| Full budget | 2,000 | -0.535 |
| adaptive_value (reference) | 667 | +0.020 |

Tripling the training budget improved performance by **+0.784 chips/round** -- a massive gain. However, even with full budget, pruned_history (-0.535) still falls far below adaptive_value (+0.020), indicating that the budget fix alone is insufficient.

### 2. Network Capacity Mismatch

With 31-dim input and 64 hidden units, the network has approximately 6,300 parameters but receives only ~20,000 training hands at 667 sessions -- a 3:1 sample-to-parameter ratio. Compare with adaptive_value: 19-dim input, 64 hidden, ~5,200 parameters, same ~20,000 hands -- a better ratio and smaller input space.

The adaptive_history agent addressed this by using 128 hidden units for its 35-dim input, nearly doubling network capacity. By keeping pruned_history at 64 hidden units while expanding the input from 19 to 31 dimensions, the network is too narrow to effectively learn from the expanded feature space.

### 3. Feature Pruning Validation

The pruning of fold count features was validated as **lossless** -- fold counts in action history are mathematically guaranteed to be zero (fold terminates the hand, so no subsequent actions are recorded). Removing 4 useless features from 16 saves 25% of the history encoding without losing any information.

### 4. Compound Failure Mode

The poor performance is a **compound** of two issues:

1. **Budget** (primary): 667 sessions was clearly insufficient for the larger 31-dim input space. The 2000-session version shows dramatic improvement.
2. **Network width** (secondary): Even at 2000 sessions, 64 hidden units for 31-dim input is too narrow. The network cannot develop the representational capacity to effectively use the history features alongside the stats.

### 5. Training Loss Profile

The training loss over 667 sessions starts around 19.2 and converges to approximately 14.6 -- still relatively high compared to adaptive_value's typical final loss around 7-8. This confirms the network has not converged and would benefit from both more training and more capacity.

---

## What Would Fix It

Based on the diagnosis, a competitive pruned_history agent would need:
1. **2000+ sessions**: The budget sweep showed monotonic improvement with no overfitting
2. **128 hidden units**: Matching the network width to the input dimensionality, as adaptive_history did for its 35-dim input
3. **Combined fix**: Both changes simultaneously, since the budget and capacity issues compound each other

The concept of pruning zero-valued features is sound and validated. The execution was undermined by infrastructure issues (wrong budget) and a conservative architecture choice (keeping parent's network width despite doubled input).

---

## Key Insight

When expanding the observation space, network width must scale proportionally -- doubling the input dimensions without increasing hidden units creates an architecture capacity mismatch that no amount of training can fully overcome.

---

## Source Files

- Agent: `src/agents/pruned_history.py`
- Trainer: `src/training/pruned_history_trainer.py`
- Diagnosis: `experiments/diagnose_pruned_extended.py`
- Diagnostic Results: `experiments/diagnose_pruned_extended_results.json`
