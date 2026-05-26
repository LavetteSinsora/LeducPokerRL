# dual_axis_repr_v2 — Dual-Axis SupCon Representation

## Research Question

Can a supervised contrastive learning (SupCon) approach with two independent label axes —
expected reward bins AND opponent hand identity — simultaneously encode both strategic value
and opponent identity in a single representation, preventing the 1D collapse seen in prior
experiments?

## Motivation: Why SupCon Instead of Joint AND/OR Logic

The prior approach (`dual_axis_repr_v1`) defined positive/negative pairs using `AND`/`OR`
logic across both labels. This is ad-hoc and suffers from severe sparsity: few anchors find
positives satisfying both conditions simultaneously within a batch.

The established approach in contrastive learning literature (**Supervised Contrastive
Learning, Khosla et al. 2020, NeurIPS**) uses per-axis SupCon losses summed together:

```
L_total = L_SupCon(reward_bin_labels) + λ_hand * L_SupCon(hand_labels) + λ_var * L_VICReg
```

Each SupCon loss independently defines its own positive set (all samples with the same label
on that axis), avoiding the AND/OR sparsity problem. The two objectives share the same
encoder, so gradients from both axes co-train a single representation.

## Two-Axis Hypothesis

A SupCon encoder trained with two independent label axes will learn a representation that:
1. Clusters states with similar terminal rewards together (reward axis)
2. Clusters states facing the same opponent hand together (identity axis)
3. Uses at least 2 embedding dimensions (avoids 1D collapse from hand_identity_repr_v1)

Without VICReg, SupCon alone can collapse to a single line (all embeddings pushed to
cluster centroids with degenerate rank-1 covariance). VICReg variance penalty prevents this
by penalizing dimensions with low standard deviation.

## Success Criteria

| Metric | Target | Reference |
|--------|--------|-----------|
| Effective dimension (80% variance) | >= 2 | hand_identity_repr_v1: 1 |
| Opponent hand linear probe accuracy | > 50% | chance: 33.3% |
| Reward Spearman ρ (pairwise distance vs \|ΔR\|) | > 0.163 | contrastive_repr_v1: 0.163 |

## Architecture

- **Encoder**: 15 → 64 → 64 → 8 (no final normalization; SupCon normalizes internally)
- **Data**: Self-play with frozen ValueBasedAgent; record (state_enc, terminal_reward, opp_hand_label)
- **Buffer**: 5000-sample replay deque
- **Batch**: 256 samples/update
- **Reward bins**: 5 bins with thresholds [-inf, -2.0, -0.5, 0.5, 2.0, +inf]
- **Losses**: SupCon(τ=0.07) for reward bins + SupCon(τ=0.07) for hand labels + VICReg variance

## Key Differences from Prior Experiments

- **vs contrastive_repr_v1**: Uses SupCon categorical loss instead of soft L1 distance; adds hand axis
- **vs hand_identity_repr_v1**: Adds reward axis SupCon; VICReg prevents 1D collapse
- **vs dual_axis_repr_v1**: Principled per-axis SupCon replaces ad-hoc AND/OR joint logic
