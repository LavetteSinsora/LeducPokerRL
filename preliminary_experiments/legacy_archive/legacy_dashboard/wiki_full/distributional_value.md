# Distributional Value Agent

> Risk-sensitive decision-making via a dual-head value/variance network -- a working architecture that reveals poker's inherent variance makes risk-aversion collapse to always folding.

| Property | Value |
|----------|-------|
| **ID** | `distributional_value` |
| **Parent** | value_based |
| **Round** | 4 |
| **Rank** | ~19 / 22 |
| **Avg Score** | -0.580 |
| **Robustness** | -1.230 |

---

## Motivation

Standard value-based agents maximize expected value E[R], but this ignores an important dimension of poker: **variance**. Two actions might have the same expected value but wildly different risk profiles. For example, a raise might win +5 or lose -5 (high variance), while a call might win +1 or lose -1 (low variance). A risk-sensitive agent could prefer the lower-variance action, trading some expected value for consistency.

The distributional value agent learns both the mean and variance of returns, then makes decisions using a **risk-adjusted value**: `V_risk(s,a) = E[V] - beta * Var[V]`. The parameter beta controls risk sensitivity -- at beta=0 the agent is risk-neutral (identical to value_based), and at higher beta values it increasingly avoids high-variance actions.

---

## Architecture

### Development History: 6 Iterations

The final architecture emerged after 6 failed or partial attempts:

| Attempt | Architecture | Outcome |
|---------|-------------|---------|
| 1-3 | Quantile regression | Diverged (3 separate attempts) |
| 4 | Shared trunk, dual output | Gradient interference between value and variance heads |
| 5 | Separate networks, shared optimizer | Variance head dominated gradients |
| 6 | **Dual-head, separate optimizers** | **Stable training** |

### Final Architecture: DualHeadModel

```
ValueNetwork(15 -> 64 -> 64 -> 1)
  - Input: 15-dim game state encoding (same as value_based)
  - Output: Scalar expected value E[V]
  - Optimizer: Adam, lr=1e-4

VarianceNetwork(15 -> 64 -> 64 -> softplus -> 1)
  - Input: 15-dim game state encoding (same input)
  - Output: Scalar variance estimate Var[V] (always non-negative via softplus)
  - Optimizer: Adam, lr=1e-4 (separate optimizer)
```

The **separate optimizers** are critical -- sharing an optimizer caused the variance gradients to destabilize the value head. The softplus activation on the variance output ensures non-negative variance predictions.

### Risk-Adjusted Decision Making

At decision time, the agent evaluates each legal action using 1-step lookahead (same as value_based), but with a risk penalty:

```
V_risk(s, a) = E[V(s')] - beta * Var[V(s')]
```

where `s'` is the successor state after taking action `a`. Higher beta penalizes high-variance outcomes more heavily.

### Observation Encoding (15 dimensions)

Same as the parent value_based agent -- no changes to the input representation.

---

## Training Configuration

| Parameter | Value |
|-----------|-------|
| Training Method | Self-play with TD(0) |
| Episodes | 20,000 |
| Value Learning Rate | 1e-4 (Adam) |
| Variance Learning Rate | 1e-4 (Adam, separate) |
| Value Loss | TD(0) MSE |
| Variance Loss | MSE on squared TD error |
| Beta (risk parameter) | 0.5 (tuned via sweep) |

The variance network is trained to predict the squared TD error: `Var_target = (r + V(s') - V(s))^2`. This is an estimate of the variance of the value estimate at each state.

---

## Tournament Results (beta=0.5)

| Opponent | Score |
|----------|-------|
| heuristic | -0.20 |
| value_based | -0.91 |
| adaptive_value | -0.74 |
| modulated_value | -1.04 |
| entropy_ac | +0.08 |
| cfr | -0.67 |

**Best matchup**: +0.08 vs entropy_ac
**Worst matchup**: -1.04 vs modulated_value

### Beta Sweep Results

| Beta | Avg Score | Robustness | Behavior |
|------|-----------|------------|----------|
| 0.0 | -0.86 | -- | Risk-neutral (equivalent to value_based) |
| 0.3 | -0.43 | -- | Mild risk aversion |
| 0.5 | **-0.37** | **-0.88** | Best average, best robustness |
| 1.0 | -1.00 | -- | Always folds (collapsed) |

The optimal beta (0.5) sits in a narrow sweet spot. Below 0.3, risk adjustment has minimal effect. Above 0.5, the agent becomes so risk-averse that it folds almost every hand.

---

## Diagnosis & Findings

### 1. Risk-Sensitivity Shifts RAISE to CALL

The primary strategic effect of risk-aversion is a systematic shift from RAISE to CALL. Raising increases pot size, which increases the variance of the final outcome. Calling keeps the pot smaller, producing lower-variance results.

At beta=0.5, approximately **70% of the agent's decisions differ from what ValueBasedAgent would choose**. The vast majority of these differences are RAISE (value_based) becoming CALL (distributional).

### 2. Poker's Inherent Variance Creates a Collapse Boundary

Leduc Hold'em has inherent outcome variance of approximately std=3-4 chips per hand. This variance is irreducible -- it comes from the random card deals, not from the agent's decisions. At high beta values, the risk penalty `beta * Var[V]` becomes so large that it dominates the expected value term for every action except FOLD (which has zero variance because it immediately ends the hand with a known loss).

This creates a **collapse boundary**: above a critical beta, the agent's risk-adjusted values make FOLD the optimal action for every hand, including strong hands that should clearly be raised. At beta=1.0, the agent folds 100% of hands.

### 3. The Narrow Viable Range

The viable range for beta is approximately [0.3, 0.6]. Below this range, risk adjustment is too weak to matter. Above it, the agent collapses to all-fold. This narrow window means the distributional approach is fragile -- it requires careful tuning of a parameter that has a sharp phase transition.

### 4. Comparison with Parent

| Metric | value_based | distributional (beta=0.5) |
|--------|------------|--------------------------|
| Decision overlap | 100% | 30% (70% differ) |
| Primary shift | -- | RAISE becomes CALL |
| Folding rate | Low | Higher |
| Avg Score | ~-0.10 | -0.37 |

The distributional agent makes fundamentally different decisions than its parent, but these differences hurt average performance. The shift from RAISE to CALL is too conservative -- in Leduc Hold'em, aggression is generally rewarded, and the variance reduction from calling does not compensate for the lost value from not raising.

---

## Assumptions & Limitations

1. **Beta sensitivity and collapse boundary**: Beta values above ~0.5 cause the agent to collapse to always-fold. Poker has inherent outcome variance of std 3-4 chips per hand from card randomness alone. Since FOLD always produces zero variance (the hand ends immediately with a known loss), any beta high enough to make the variance penalty significant will make FOLD dominate all other actions. The viable beta range is narrow: approximately [0.3, 0.6]. This makes the distributional approach fragile -- a sharp phase transition separates useful risk-aversion from complete collapse.

2. **Mean-only training**: The variance network is trained separately from the value network using separate optimizers to prevent gradient interference. Critically, risk-sensitivity is applied only at decision time (evaluation), NOT during training. During training, exploration uses the mean value only (standard Boltzmann on E[V]). This means the training trajectories do not reflect the risk-adjusted policy, creating a train-eval mismatch. The agent trains as if risk-neutral but acts as if risk-averse.

3. **Quantile regression instability**: The original architecture (attempts 1-3) used bootstrapped quantile regression to learn the full return distribution. All three attempts diverged: quantile taus at the boundaries push outputs to extreme values, and the "distribution" being learned via self-play bootstrapping is not mathematically well-defined (the bootstrap target's distribution shifts as the network updates). The dual-head mean+variance architecture was needed as a more stable alternative, though it captures less distributional information (only first two moments vs. the full distribution).

---

## Key Insight

Risk-sensitivity works in theory but collapses in practice for poker: the game's inherent variance (std of 3-4 chips per hand from random deals) is so large relative to strategic variance that even moderate risk-aversion parameters push the agent toward always folding -- the only zero-variance action.

---

## Source Files

- Agent: `src/agents/distributional_value.py`
- Trainer: `src/training/distributional_value_trainer.py`
