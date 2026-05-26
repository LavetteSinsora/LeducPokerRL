# N-Step Value Agent

> Value network trained with n-step TD returns instead of TD(0), trading bootstrap bias for Monte Carlo variance.

| Property | Value |
|----------|-------|
| **ID** | `nstep_value` |
| **Parent** | ValueBasedAgent (`value_based`) |
| **Round** | 2 |
| **Rank (R2)** | #6 / 12 |
| **Rank (R3)** | #9 / 17 |
| **Avg Score (R2)** | +0.28 |
| **Robustness (R2)** | -1.08 |
| **Avg Score (R3)** | +0.072 |
| **Robustness (R3)** | -1.007 |

## Motivation

ValueBasedAgent uses TD(0) for training -- each value prediction bootstraps from the immediate next state's value estimate. TD(0) has low variance but high bias (the bootstrap target is itself an approximation). The hypothesis was that using n-step returns (n=3) would provide cleaner gradient signal by reaching closer to the actual terminal reward before bootstrapping, reducing the bias inherent in 1-step lookahead.

In theory, n-step returns sit on the bias-variance spectrum between TD(0) (n=1, low variance, high bias) and Monte Carlo (n=infinity, zero bias, high variance). The hope was that n=3 would hit a sweet spot.

## Architecture

NStepValueAgent inherits directly from ValueBasedAgent with **zero architectural changes**:

```python
class NStepValueAgent(ValueBasedAgent):
    """Value agent trained with n-step returns instead of TD(0)."""
    pass
```

The modification is entirely in the trainer. The underlying network is:

```
ValueNetwork:
  Linear(15 -> 64) -> ReLU -> Linear(64 -> 64) -> ReLU -> Linear(64 -> 1)
```

- **Input**: 15 dimensions (3 hand + 4 board + 2 pot + 6 features)
- **Hidden**: 64 units, 2 layers
- **Output**: 1 scalar value V(s)

## Training Configuration

| Parameter | Value |
|-----------|-------|
| Episodes | 20,000 |
| Learning Rate | 1e-4 |
| Batch Size | 30 |
| **N-Steps** | **3** |
| Eval Interval | Every 50 episodes |
| Eval Games | 100 vs heuristic |
| Optimizer | Adam |
| Loss | MSE |
| Exploration | Boltzmann (temperature=1.0) |

### The N-Step Return

For a per-player post-action chain of length L, the target at timestep t is:

```
if t + n >= L:  target = terminal_reward    (no bootstrapping)
if t + n <  L:  target = V(s_{t+n})         (bootstrap from n steps ahead)
```

Since Leduc Hold'em games produce chains of 2-4 steps and n=3, most transitions use the actual terminal reward directly. There is no discounting (gamma=1) and rewards are terminal-only.

## Tournament Results

### Round 2 (12-agent round-robin, 500 rounds/matchup)

| Opponent | Score |
|----------|-------|
| heuristic | +0.07 |
| value_based | -0.95 |
| adaptive_value | -0.87 |
| aux_value | **+1.67** |
| actor_critic | +0.59 |
| history_value | +0.90 |
| decay_adaptive | **+1.56** |
| entropy_ac | -0.57 |
| pop_adaptive | +0.54 |
| adaptive_history | -0.39 |
| target_value | +0.55 |

### Round 3 (17-agent round-robin, 1000 rounds/matchup)

| Stat | Value |
|------|-------|
| Average | +0.072 |
| Worst | -1.027 |
| Best | +1.064 |
| Std | 0.720 |
| Robustness | -1.007 |

## Diagnosis & Findings

A comprehensive diagnostic experiment (`diagnose_nstep_results.json`) swept across n=1 through n=4 and Monte Carlo (MC), training 5,000 episodes each and evaluating against heuristic.

### N-Step Loss Comparison

| Method | Loss Mean | Loss Std | Rolling Var (Mean) | Rolling Var (Last 10) | Final Score |
|--------|-----------|----------|--------------------|-----------------------|-------------|
| TD(0) / n=1 | 17.74 | 3.75 | 13.44 | 17.68 | -0.34 |
| n=2 | 35.45 | 7.30 | 51.77 | 108.03 | -1.65 |
| **n=3** | **46.22** | **10.98** | **111.70** | **110.63** | **-0.76** |
| n=4 / MC | 49.85 | 11.98 | 134.33 | 129.28 | -0.91 |

### Critical Finding: N=3 is Pure Monte Carlo in Leduc

With typical chain lengths of 2-4 steps:
- At n=3, the condition `t + 3 >= L` is satisfied for virtually all transitions
- 99.1% of TD targets end up using the terminal reward directly
- This means n=3 is effectively pure Monte Carlo, not a hybrid

The consequences are severe:
1. **Loss increases monotonically with n**: From 17.7 (n=1) to 49.8 (n=4/MC) -- a 2.8x increase
2. **Variance increases monotonically**: Rolling variance goes from 13.4 to 134.3 -- a 10x increase
3. **No sweet spot exists**: There is no intermediate n that outperforms TD(0) in this domain

### Why TD(0) Wins in Short Games

In Leduc Hold'em, bootstrapping from V(s_{t+1}) serves as implicit **temporal smoothing**. The value network aggregates information from many training episodes, providing a stable regression target even though any individual game's outcome is noisy. Removing this smoothing (by using MC returns) exposes the optimizer to the full variance of terminal rewards.

### Training Curves

The eval history for n=3 shows unstable learning:

| Episode | Score vs Heuristic |
|---------|--------------------|
| 500 | -0.65 |
| 1000 | -0.445 |
| 1500 | -0.48 |
| 2000 | -1.185 |
| 2500 | -0.83 |
| 3000 | -0.58 |
| 3500 | -0.41 |
| 4000 | -1.16 |
| 4500 | -0.425 |
| 5000 | -1.58 |

Performance oscillates wildly with no clear convergence trend, confirming the high-variance gradient signal.

## Key Insight

The bias-variance tradeoff of n-step returns does not favor n > 1 in very short games. When game episodes are only 2-4 steps long, n=3 degenerates into pure Monte Carlo, losing the temporal smoothing benefit that makes TD(0) effective. In Leduc Hold'em, TD(0)'s "biased" bootstrapping is actually a feature, not a bug.

## Source Files

- Agent: `src/agents/nstep_value.py`
- Trainer: `src/training/nstep_value_trainer.py`
- Parent Agent: `src/agents/value_based.py`
- Diagnosis: `experiments/diagnose_nstep_results.json`
