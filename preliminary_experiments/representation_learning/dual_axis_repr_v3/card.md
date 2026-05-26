# dual_axis_repr_v3 — Hybrid L1 Soft-Distance + SupCon Hand

## Research Question
Does combining the exact reward soft-distance loss from `contrastive_repr_v1` with the SupCon hand-identity loss from `dual_axis_repr_v2` produce a representation that achieves full fidelity on BOTH axes simultaneously?

## Motivation

Prior experiments established a clear progression:

| Metric | reward-only (v1) | hand-only | dual SupCon (v2) |
|---|---|---|---|
| Effective dim (80%) | 3 | 1 | 2 |
| Opp hand accuracy | ~33% | 62.8% | 63.3% |
| Reward Spearman ρ | **0.163** | — | 0.118 |

`dual_axis_repr_v2` matched hand accuracy but the reward Spearman ρ dropped from 0.163 to 0.118. The root cause: SupCon uses discrete reward bins, which clusters within-bin states but does NOT enforce the continuous metric ordering between bins that the L1 soft-distance loss provides.

**The fix**: Keep the L1 reward soft-distance loss exactly as in `contrastive_repr_v1`, and ADD the hand SupCon loss on top. This preserves the continuous metric structure in the reward axis while adding hand identity separability.

## Loss Design

```
L_total = L_L1_reward + λ_hand * L_SupCon(hand) + λ_var * L_VICReg
```

- **`L_L1_reward`**: Exact L1 soft distance correlation from `contrastive_repr_v1` with per-batch β auto-calibration
- **`L_SupCon(hand)`**: Supervised contrastive loss over opponent hand labels (J/Q/K), temperature=0.07
- **`L_VICReg`**: Variance regularizer (prevents dimensional collapse)

## Hypothesis
v3 will exceed reward Spearman ρ > 0.163 AND maintain opp hand accuracy > 63% because:
1. The L1 loss preserves continuous metric ordering (not just bin membership)
2. The hand SupCon loss adds separability for opponent hand identity
3. The two objectives operate on different structural properties and should complement rather than compete

## Success Criteria
- Effective dim (80%) ≥ 3
- Opp hand accuracy ≥ 63%
- Reward Spearman ρ > 0.163

## Hyperparameters
- λ_hand = 1.0 (default; try 0.5 or 0.1 if unstable)
- λ_var = 0.1 (same as v2, stabilizes training)
- SupCon temperature = 0.07 (standard)
- Encoder: 15 → 64 → 64 → 8 (unchanged)
- Optimizer: Adam, lr=1e-4
- Buffer: deque(maxlen=5000)
- Batch size: 256
- Episodes: 20000
