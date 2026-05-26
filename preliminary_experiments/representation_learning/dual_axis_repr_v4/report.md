# dual_axis_repr_v4: EMA-Normalized Dual-Axis Representation — Report

## Results Summary

| Metric | reward-only | hand-only | dual SupCon (v2) | dual hybrid (v3) | **dual EMA-norm (v4)** |
|---|---|---|---|---|---|
| Effective dim (80%) | 3 | 1 | 2 | 2 | **4** |
| Opp hand accuracy | ~33% | 62.8% | 63.3% | 67.8% | **38.3%** |
| Reward Spearman ρ | 0.163 | — | 0.118 | 0.083 | **0.672** |
| Reward bin accuracy | — | — | 55.4% | 42.9% | **23.8%** |

## Research Question: Did EMA Normalization Fix the Scale Imbalance?

**Partially yes — but overcorrected in favor of reward.**

EMA normalization did fix the scale imbalance problem, but in an unexpected direction: instead of equalizing both axes, it inverted the dominance. The reward loss won the gradient war instead of the hand loss.

## What Happened

### The EMA Convergence Problem

The reward loss (`L_reward`) converges to nearly zero during training because the SoftDistanceL1Loss auto-calibrates beta each batch — as the encoder learns to match the distance structure, the loss shrinks. By convergence:

- `L_reward_raw ≈ 2.4e-6` (the encoder learned to match reward distances very well)
- `L_hand_raw ≈ 5.53` (hand loss stays near constant — hard classification problem)
- `EMA_reward ≈ 2.5e-6`
- `EMA_hand ≈ 5.53`

So the normalized losses are:
- `L_reward / EMA_reward ≈ 1.0` (normalized to unit scale)
- `L_hand / EMA_hand ≈ 1.0` (also normalized to unit scale)

This looks balanced! But critically, the **reward loss has already converged to near-zero**. Once EMA tracks a near-zero value, any tiny fluctuation in `L_reward` produces a large normalized gradient. The reward axis dominates the gradient signal because it has a steep loss surface relative to its magnitude — the beta auto-calibration creates a well-conditioned optimization landscape for reward ordering.

### The Result: Reward Dominance

- Reward Spearman ρ = **0.672** — dramatically better than any prior version (v3: 0.083, reward-only: 0.163)
- The embedding space has learned strong reward distance structure
- Effective dimension = 4 (up from 2 in v2/v3) — richer representation
- Hand accuracy = **38.3%** — barely above chance (33%), down from 67.8% in v3

The EMA normalization overcorrected: by normalizing both losses, it gave the reward loss — which has a more favorable optimization landscape — a greater effective influence.

## Key Insight: EMA Normalization is Not Gradient Equalization

The EMA normalizer makes the **loss values** equal in magnitude, but not the **gradient norms**. The reward loss has much larger gradients per unit of loss value (because beta auto-calibration creates a well-conditioned metric learning problem), while SupCon at convergence produces smaller per-unit gradients due to the logistic saturation in the softmax.

To truly equalize gradient contributions, one would need to normalize by the **gradient norm** of each loss, not the loss value.

## Positive Finding: Reward Structure is Recoverable

v4 provides the strongest evidence yet that the 8-dim encoder CAN learn reward structure: Spearman ρ = 0.672 far exceeds both the reward-only baseline (0.163) and any dual-axis version. The EMA normalization — even though it accidentally favored reward — demonstrates that the hand SupCon objective was genuinely suppressing reward learning in v2/v3.

## Failure to Meet Success Criteria

- Reward Spearman ρ > 0.163: **ACHIEVED** (0.672)
- Opp hand accuracy ≥ 63%: **FAILED** (38.3%)

Both criteria were not simultaneously satisfied; v4 traded hand discrimination for reward structure.

## Implications for v5

The v4 outcome reframes the research question. Rather than normalizing losses to equal scale, future work should:

1. **Explicit gradient balancing**: Normalize by gradient norm (e.g., GradNorm method) rather than loss magnitude
2. **Asymmetric weighting**: Use `lambda_hand >> 1.0` (e.g., 100× or 1000×) to counteract the reward loss's natural gradient advantage
3. **Separate representation heads**: Train reward and hand axes on different embedding subspaces, preventing competition
4. **Stop EMA normalization when reward converges**: Once `L_reward < threshold`, switch to fixed weight to prevent reward overdominating

## Training Details

- Episodes: 20,000
- Batch size: 256
- Learning rate: 1e-4
- EMA alpha: 0.99
- Lambda_hand: 1.0, Lambda_var: 0.1
- Training time: 81.9s

## Conclusion

EMA normalization solved the v3 problem (hand dominated reward) but introduced the opposite problem (reward dominated hand). The fundamental issue is that loss magnitude normalization ≠ gradient contribution equalization. The strong Spearman ρ = 0.672 is a useful finding — it shows the encoder architecture is capable of strong reward structure — but a different equalization mechanism is needed to achieve dual-axis learning simultaneously.
