# dual_axis_repr_v4: EMA-Normalized Dual-Axis Representation

## Research Question

Can EMA-based loss normalization solve the 2,600× scale imbalance between the L1 reward soft-distance loss and the SupCon hand loss, allowing both axes to contribute equally to the representation?

## Motivation

v3 used `L_total = L_L1_reward + 1.0 * L_SupCon(hand) + 0.1 * L_VICReg`, but at convergence:
- `L_hand ≈ 5.26` vs `L_reward ≈ 0.002` — a **2,600× gap**
- The optimizer received almost exclusively hand gradients; reward Spearman ρ collapsed to 0.083
- Hand accuracy improved to 67.8% because it faced zero competition from the reward loss

The fundamental problem: raw loss values are at incomparable scales. Adding them directly means the SupCon loss completely dominates gradient flow.

## Fix: EMA Normalization

Normalize each loss by its running exponential moving average (EMA) magnitude before combining:

```
L_total = (L_reward / ema_reward) + λ_hand * (L_hand / ema_hand) + λ_var * L_var
```

The EMA tracks the recent average magnitude of each loss, making the normalized contributions unit-scale and directly comparable. With λ_hand = 1.0, both losses should push with equal gradient magnitude each step.

## Hypothesis

EMA normalization will rebalance gradient contributions from the two losses, recovering reward structure (Spearman ρ > 0.163) while maintaining meaningful hand discrimination (accuracy ≥ 63%).

## Success Criteria

- Reward Spearman ρ > 0.163 (beat reward-only baseline)
- Opp hand accuracy ≥ 63% (maintain dual-axis discrimination)
- Final EMA values should converge to stable magnitudes (confirming normalization worked)

## Architecture

Identical to v3: 15→64→64→8 encoder, same observation encoding.

## Key Implementation Detail

EMA bias correction prevents instability in early training steps. Without it, the EMA starts near zero and produces very large normalized values. The bias correction `ema / (1 - α^t)` fixes this for small t.

VICReg loss is NOT normalized — it is a regularizer at a naturally compatible scale.
