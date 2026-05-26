# value_based_repr_analysis

**Question**: Does a TD(0)-trained value network's internal 64-dim representation
organize game states by reward proximity — and does this emerge as well or better
than explicitly contrastive-trained encoders?

**Metrics**:
- Scalar Spearman ρ: correlation between V(s) output and true terminal reward R
  (measures value function accuracy)
- Hidden Spearman ρ: pairwise L2 distance in 64-dim penultimate layer vs pairwise |ΔR|
  (measures reward-metric structure; same metric as contrastive experiments)
- Raw 15-dim Spearman ρ: baseline using raw input features directly

**Baseline comparisons**:
- contrastive_repr_v1 (L1 soft-distance, 8-dim): Hidden ρ = 0.163
- dual_axis_repr_v4 (EMA-normalized, 8-dim): Hidden ρ = 0.672

The key insight is whether value-function training implicitly creates reward-metric
structure as a byproduct, or whether explicit contrastive objectives are necessary
to achieve it.
