# dual_axis_repr_v3 Report — Hybrid L1 Soft-Distance + SupCon Hand

**Date**: 2026-03-11
**Run**: `outputs/dual_axis_repr_v3/run_default` — 20,000 episodes, 61s

## Full Comparison Table

| Metric | reward-only (v1) | hand-only | dual AND/OR (v1) | dual SupCon (v2) | **dual hybrid (v3)** |
|---|---|---|---|---|---|
| Effective dim (80%) | 3 | 1 | 2 | 2 | **2** |
| Effective dim (90%) | — | — | — | — | **2** |
| Opp hand accuracy | ~33% | 62.8% | 53.8% | 63.3% | **67.8%** |
| Reward Spearman ρ | 0.163 | — | 0.107 | 0.118 | **0.083** |
| Reward bin accuracy | — | — | — | 55.4% | **42.9%** |

## Key Findings

### 1. Did L1 soft-distance recover reward Spearman ρ above 0.163?
**No.** Reward Spearman ρ = 0.083, which is worse than both v1 (0.163) and v2 (0.118). The L1 reward signal is present and active (final L_reward ≈ 0.0024) but the hand SupCon loss dominates by ~2000x in absolute magnitude (L_hand ≈ 5.26 vs L_reward ≈ 0.002 at convergence). This extreme scale imbalance means the optimizer is almost entirely driven by the hand objective, suppressing the metric structure the reward loss is trying to impose.

### 2. Did hand SupCon maintain accuracy above 63%?
**Yes, and exceeded it.** Opp hand accuracy = 67.8%, which is higher than v2's 63.3%. The hand SupCon loss alone is driving the representation with minimal competition from the L1 reward term. The hand objective benefited from the undivided capacity.

### 3. Did the two objectives compete or complement each other?
**They competed, with hand winning decisively.** The PCA variance breakdown is revealing:
- PC1: 50.6% variance
- PC2: 40.3% variance
- PCs 3–8: 9.1% combined

Only 2 components explain 90% of variance. This means the 8-dimensional embedding is effectively 2D — the same as v2. The representation is specialized entirely for hand identity, not for reward metric structure.

The reward bin accuracy (42.9%) is also lower than v2 (55.4%), confirming that the continuous L1 loss actually organized reward information *less* effectively than SupCon reward bins for linear probing. The L1 loss encourages pairwise metric ordering but apparently this doesn't translate to linearly separable reward bins when the hand loss dominates.

### 4. Did effective dimensionality expand to 3+?
**No.** Effective dim (80%) = 2, same as v2. With the hand loss dominating, there are two strong modes of variation (J/Q/K hand identity structure), while the reward signal is crowded out.

## Diagnosis: Scale Imbalance Problem

The root cause of v3's underperformance on the reward axis is loss scale mismatch:
- At convergence: L_hand ≈ 5.26, L_reward ≈ 0.002
- Effective reward contribution: 0.002 / 5.26 ≈ 0.04% of total gradient signal

The L1 loss auto-calibrated β correctly (it works), but the SupCon cross-entropy loss operates on normalized cosine similarities and naturally produces losses in the 0–log(N) range (~5-6 for N=256), while the L1 squared distance loss converges to very small values as the metric is learned.

**Potential fixes for a v4 experiment**:
1. Normalize both losses to the same scale before combining (e.g., detach and scale)
2. Use a much larger λ_reward (e.g., λ_reward=100 or λ_reward=1000) to compensate
3. Separate the two axes into different subspaces (e.g., first 4 dims for reward, last 4 for hand) with no cross-task gradient interference
4. Use gradient normalization per-task before combining

## Training Dynamics

The VICReg loss (V ≈ 0.99) indicates that embedding dimensions are never reaching target std=1.0 — the representation is being forced into narrow distributions along most dimensions. This is consistent with an effectively 2-dimensional representation despite being nominally 8-dimensional.

## Configuration

```
lambda_hand = 1.0
lambda_var = 0.1
lr = 1e-4
batch_size = 256
episodes = 20000
SupCon temperature = 0.07
Encoder: 15 → 64 → 64 → 8
```

## Conclusion

v3 succeeds on the hand axis (67.8%, new best) but fails on the reward axis (ρ=0.083, worst of all versions). The hybrid approach did not achieve "both at full fidelity" simultaneously. The scale imbalance between SupCon (natural log-scale loss ~5) and L1 soft-distance (converges toward 0 as learned) causes the hand objective to dominate gradient flow completely.

The lesson: combining losses from different loss families without explicit scale balancing or subspace separation leads to one loss dominating. A v4 should either (a) use matched loss scales, or (b) separate the representation into hand-only and reward-only subspaces.
