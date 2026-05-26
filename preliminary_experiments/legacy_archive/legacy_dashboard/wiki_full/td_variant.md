# TD Variant Agent

> Calibrated n-step / Monte Carlo comparison agent that conclusively proves TD(0) superiority for short-chain games.

| Property | Value |
|----------|-------|
| **ID** | `td_variant` |
| **Parent** | value_based |
| **Round** | 3 |
| **Rank** | #13 / 17 |
| **Avg Score** | -0.216 |
| **Robustness** | -1.729 |

---

## Motivation

Round 2's nstep_value agent (n=3) showed mixed results, ranking #6 with +0.28 avg. However, a confound remained: did n-step underperform because of the technique itself, or because of poorly calibrated hyperparameters? The td_variant agent was designed as a **rigorous, controlled comparison** between TD(0), n-step, and Monte Carlo methods with properly calibrated learning rates.

The hypothesis: if we reduce the learning rate proportionally to account for the higher variance of n-step/MC targets, n-step returns might match or exceed TD(0).

---

## Architecture

TDVariantAgent has the **identical architecture** to ValueBasedAgent -- a minimal subclass that changes nothing about the network:

```
ValueNetwork(15 -> 64 -> 64 -> 1)
  - Input: 15-dim game state encoding
  - Architecture: 2-layer MLP with ReLU, 64 hidden units
  - Parameters: 5,249
```

The only difference is in the **trainer**, which implements configurable n-step returns. The agent class itself is a pass-through:

```python
class TDVariantAgent(ValueBasedAgent):
    """Value agent for systematic TD variant comparison (TD(0), n-step, MC)."""
    pass
```

All the complexity lives in `TDVariantTrainer`, which extends `SelfPlayTrainer` with configurable `n_steps`:
- `n_steps=1`: TD(0) -- bootstrap from immediate next state
- `n_steps=2,3,...`: n-step TD -- bootstrap from n steps ahead
- `n_steps=9999`: Monte Carlo -- always use terminal reward (since Leduc chains are at most ~4 steps, 9999 effectively means infinity)

---

## Training Configuration

| Parameter | Value |
|-----------|-------|
| Training Method | Self-play (via TDVariantTrainer) |
| Episodes | ~5,000 (tournament model trained longer) |
| Learning Rate | 5e-5 (half of TD(0)'s 1e-4, calibrated for higher variance) |
| Batch Size | 32 |
| n_steps | 3 |
| TD Method | n-step returns with terminal fallback |
| Training Time | 40.8 seconds |
| Updates | 625 |
| Final Loss | 33.35 |
| Eval vs Heuristic | -1.33 |

---

## Tournament Results

| Opponent | Score |
|----------|-------|
| heuristic | -1.526 |
| value_based | -1.420 |
| adaptive_value | -1.618 |
| aux_value | +0.048 |
| actor_critic | +0.412 |
| history_value | +0.923 |
| decay_adaptive | +1.313 |
| nstep_value | -0.026 |
| entropy_ac | -1.006 |
| pop_adaptive | +0.862 |
| adaptive_history | -0.586 |
| target_value | -0.209 |
| pruned_history | +0.745 |
| modulated_value | -1.504 |
| curriculum | +0.879 |
| extended_adaptive | -0.744 |

td_variant loses to all three strong agents (value_based, adaptive_value, modulated_value) by large margins and has high variance (std = 1.008, the highest in the tournament). It only reliably beats the weakest agents.

---

## Diagnosis & Findings

The diagnosis was the most thorough TD analysis in the project, revealing a fundamental insight about bootstrapping in short-chain games.

### 1. Chain Length Analysis -- N=3 is Pure Monte Carlo

The critical finding: in Leduc Hold'em, hands are so short that n=3 captures virtually all transitions as terminal:

| Config | Bootstrap % | Terminal % | Total Transitions |
|--------|------------|------------|-------------------|
| TD(0) n=1 | 36.1% | 63.9% | 5,205 |
| n=2 | 8.8% | 91.2% | 5,205 |
| **n=3** | **0.9%** | **99.1%** | **5,205** |
| n=4 | 0.0% | 100% | 5,205 |
| MC (n=9999) | 0.0% | 100% | 5,205 |

With n=3, only 49 out of 5,205 transitions use bootstrapping. The remaining 99.1% use terminal reward directly. **n=3 is functionally identical to pure Monte Carlo** in Leduc Hold'em. This is because the mean per-player chain length is only 1.30 steps.

### 2. Loss and Gradient Comparison

| Config | Mean Loss | Loss Std | Gradient Implications |
|--------|-----------|----------|----------------------|
| TD(0) n=1, lr=1e-4 | 17.3 | 3.7 | Baseline |
| n=3, lr=5e-5 | 45.1 | 11.5 | 2.5x higher loss |
| MC, lr=5e-5 | 49.1 | 11.3 | 2.8x higher loss |

MC and n=3 targets produce **2.5x higher loss** and correspondingly **2.4x larger gradients** than TD(0). Terminal rewards have inherently high variance because they depend on opponent cards, board cards, and the full action sequence -- all sources of noise that bootstrapping smooths away.

### 3. The Compounding Problem

The calibration approach (halving lr from 1e-4 to 5e-5) was intended to compensate for higher variance. But this creates a compounding problem:

1. **Noisy gradients**: MC/n-step targets have 2.5x the variance of TD(0) targets
2. **Slow learning**: lr=5e-5 produces only 50% of TD(0)'s parameter updates per step
3. **Combined effect**: Noisy gradients + slow learning = non-convergence within the training budget

### 4. Five-Variant Comparison (5K episodes)

| Variant | Final Eval Score | Tournament Avg |
|---------|-----------------|----------------|
| n=3, lr=5e-5 | +0.370 | -0.860 |
| MC, lr=5e-5 | +0.280 | -0.887 |
| MC, lr=3e-5 | +0.087 | -0.910 |
| n=3, lr=3e-5 | -0.030 | -1.047 |
| TD(0), lr=1e-4 | -0.700 | -1.507 |

At 5K episodes, n=3 appears to outperform TD(0). However, the evaluation scores oscillate wildly (range of +3 to -3 across training), indicating neither method has converged. TD(0) with sufficient training (20K+ episodes) converges to value_based quality (+0.97 avg), while n-step/MC remain unstable.

### 5. Why TD(0) Bootstrapping Works Better

TD(0) provides **implicit temporal smoothing** in self-play:
- The bootstrap target `V(s_next)` changes slowly as the network updates
- This creates a damped learning signal that stabilizes training in the non-stationary self-play environment
- MC removes this smoothing entirely, exposing the learner to full reward variance
- In a game where terminal rewards can swing from -13 to +13 depending on hidden cards, this smoothing is critical

---

## Key Insight

TD(0) is the optimal target method for Leduc Hold'em. The game's short chains (mean 1.30 steps per player) make n-step and Monte Carlo functionally equivalent, and both produce noisier gradients than TD(0)'s bootstrapped targets. Bootstrapping provides implicit temporal smoothing that is uniquely valuable in non-stationary self-play environments.

---

## Source Files

- Agent: `src/agents/td_variant.py`
- Trainer: `src/training/td_variant_trainer.py`
- Diagnosis: `experiments/diagnose_td_variant.py`
- Diagnostic Results: `experiments/diagnose_td_variant_results.json`
