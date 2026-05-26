# dual_axis_repr_v1

## Research Question

Can a contrastive encoder trained with positive/negative pairs defined by BOTH expected reward AND opponent hand identity simultaneously learn a richer, multi-axis representation than either supervision signal alone?

## Motivation: The Gap Between Prior Experiments

### contrastive_repr_v1 (reward-only contrastive)
- Supervision: terminal reward similarity
- Result: 3 effective dimensions, smooth reward gradient in embedding space
- Gap: **No opponent hand footprint** (silhouette = -0.029, essentially random)

### hand_identity_repr_v1 (hand-only contrastive)
- Supervision: opponent hand identity (J/Q/K) via triplet loss
- Result: 1 effective dimension (PC1 = 99.76% variance), 62.8% linear probe accuracy
- Gap: **Collapses to 1D ordinal axis** (J < Q < K), loses value structure

Neither experiment produces a representation that captures both strategic value geometry AND opponent hand identity simultaneously. An agent that can infer both dimensions from its state embedding would have richer information for downstream policy learning.

## Hypothesis

The joint positive/negative pair construction forces the encoder to cluster states that are simultaneously similar in strategic outcome AND opponent identity. This prevents the 1D collapse seen in hand_identity_repr_v1 (because reward structure requires multiple dimensions to represent the full reward geometry) while adding opponent hand structure missing from contrastive_repr_v1 (because negative pairs include same-reward / different-hand pairs).

The key insight: a state that is similar in reward but differs in opponent hand should be pushed away. This creates cross-axis pressure that cannot be resolved by collapsing to any single ordinal dimension.

## Pair Construction

**Positive pair (i, j):**
- `|R_i - R_j| < reward_thresh` (similar expected reward) AND
- `hand_i == hand_j` (same opponent hand)

**Negative pair (i, j):**
- `|R_i - R_j| > reward_margin` (very different reward) OR
- `hand_i != hand_j` (different opponent hand)

An "ignore zone" exists when `reward_thresh <= |ΔR| <= reward_margin` AND hands differ — these pairs are neither strictly positive nor negative and are excluded from the loss.

## Success Criteria

| Criterion | Target | Baseline |
|---|---|---|
| Effective dimension (80% PCA) | ≥ 2 | 1 (hand_identity_repr_v1) |
| Linear probe accuracy (opp hand) | > 50% | ~33% (contrastive_repr_v1) |
| Reward Spearman ρ | > 0.163 | 0.163 (contrastive_repr_v1) |

All three criteria must be met to claim that dual-axis construction successfully encodes both axes simultaneously.

## Hyperparameters

- Architecture: 15 → 64 → 64 → 8 (identical to prior experiments)
- Temperature: 0.5
- reward_thresh: 0.5 (Leduc rewards ∈ [-4, +4])
- reward_margin: 1.5 (creates ignore zone [0.5, 1.5])
- lambda_var: 0.1 (VICReg variance regularizer to prevent 1D collapse)
- LR: 1e-4, Adam
- Batch: 256, Buffer: 5000, Episodes: 20000
