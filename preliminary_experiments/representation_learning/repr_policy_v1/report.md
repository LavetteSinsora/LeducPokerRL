# repr_policy_v1: Policy Learning on Contrastive Representation — Report

## Setup

- **Encoder checkpoint**: `outputs/contrastive_repr_v1/encoder_l1.pt` (L1 contrastive loss, 20k episodes)
- **Training episodes**: 20,000 per variant
- **Optimizer**: Adam, lr=1e-4, batch_size=32
- **Reward normalization**: enabled (subtract batch mean, divide by std+eps)
- **Final evaluation**: 1,000 rounds vs HeuristicAgent

## Results

| Variant   | Architecture              | Avg Chips vs Heuristic | Notes                        |
|-----------|---------------------------|------------------------|------------------------------|
| baseline  | 15 → 64 → 64 → 3         | **-0.942**             | Raw features, all trainable  |
| frozen    | 8-dim repr → 64 → 64 → 3 | **-0.581**             | Encoder frozen               |
| finetune  | 8-dim repr → 64 → 64 → 3 | **-1.330**             | Encoder unfrozen             |

## Key Findings

### Does repr-based policy beat vanilla PG?

**Frozen encoder wins.** The frozen-encoder variant (-0.581) outperforms the vanilla baseline
(-0.942) by approximately **0.36 chips/round** — a meaningful improvement in Leduc Hold'em
where the average pot is ~6-8 chips.

### Is frozen better or worse than finetune?

**Frozen beats finetune substantially** (-0.581 vs -1.330). The finetune variant performed
worst of all three, underperforming the vanilla baseline.

### Training dynamics

All variants showed high variance during training (swings of ±1-2 chips/round between
consecutive eval points at 500-episode intervals). This is characteristic of REINFORCE with
reward normalization — the gradient estimates are high-variance at episode-level.

The frozen variant showed a moderate positive peak at episode 3,500 (+0.32) and maintained
higher average performance than baseline throughout mid-training (episodes 8,000–16,000).

## Mechanistic Hypotheses

### Why frozen > baseline?

The contrastive encoder was trained to cluster states by strategic outcome (L1 reward-based
contrastive loss). The 8-dim representation likely captures the high-level strategic
distinctions that matter for policy learning (hand strength relative to board, pot odds,
opponent pressure) while discarding noisy low-level features. REINFORCE operating on this
smoother manifold receives less-noisy gradient signal per episode, leading to more stable
and effective policy learning.

Additionally, the lower dimensionality (8 vs 15 input dims) reduces the search space for
the policy head, which has roughly equal parameter counts in both architectures — so the
representation quality advantage dominates.

### Why finetune < frozen and < baseline?

Finetuning the encoder jointly with the REINFORCE policy gradient is problematic for several
reasons:

1. **Reward signal too noisy for end-to-end training**: REINFORCE provides episode-level
   scalar rewards. The gradient through the encoder must back-propagate through the entire
   game trajectory, which is extremely high-variance. The encoder loses its contrastive
   structure before it can be replaced by anything useful.

2. **Catastrophic forgetting**: The contrastive encoder learned a structurally meaningful
   representation over many gradient steps with a purpose-built loss (L1 contrastive). One
   pass of noisy REINFORCE gradients can quickly corrupt this structure, leaving the agent
   with neither the original representation quality nor a policy-aligned alternative.

3. **Conflicting objectives**: The contrastive pretraining objective (make similar-outcome
   states cluster) is not the same as the REINFORCE objective (make high-reward actions more
   likely). Jointly optimizing both via a single reward signal does not resolve this tension.

## Conclusion

The hypothesis is **partially confirmed**: the frozen-encoder variant matches the success
criterion (avg_chips ≥ baseline: -0.581 > -0.942). However, the fine-tuned variant fails,
suggesting the contrastive representation is best used as a **fixed feature extractor** for
policy learning, not as a jointly-optimized module.

This points to a productive next direction: a two-phase approach where the encoder is first
trained contrastively (Phase 1) and then a policy head is trained on top of the frozen
representation (Phase 2), potentially with a larger policy head or higher learning rate since
the feature space is already well-organized.
