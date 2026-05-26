# repr_policy_v1 — Frozen v5 Encoder Evaluation

**Date**: 2026-03-12
**Checkpoint used**: `outputs/dual_axis_repr_v5/run_default/encoder.pt`
**Training**: 20,000 episodes, REINFORCE, batch_size=32, lr=1e-4, self-play
**Final evaluation**: 1,000 rounds vs HeuristicAgent

---

## Results Summary

| Variant | Encoder | Reward ρ | Opp hand acc | Avg chips vs heuristic |
|---|---|---|---|---|
| Vanilla REINFORCE | none (raw 15-dim) | — | — | -0.942 |
| Frozen L1 (v1) | contrastive_repr_v1 | 0.163 | ~33% | **-0.581** |
| Frozen v4 | dual_axis_repr_v4 | 0.672 | 38.3% | -0.748 |
| **Frozen v5** | dual_axis_repr_v5 | 0.536 | **65.2%** | -0.777 |

---

## Training Dynamics

Mid-training evaluations (every 500 episodes, 200-game windows) showed high variance:

- Best mid-training: **-0.11** (episode 10,000)
- Worst mid-training: **-1.40** (episode 1,500)
- Final 1,000-round eval (greedy): **-0.777**

The agent shows unstable learning — oscillating between near-competitive and heavily negative performance throughout the 20,000-episode run. This suggests the frozen encoder features are usable but the policy head is not converging reliably.

---

## Key Findings

### 1. v5 is NOT the best policy encoder — v1 remains the winner

Despite v5 having higher reward ρ (0.536 vs 0.163) and dramatically better hand accuracy (65.2% vs ~33%), v1 still produces the best downstream policy (-0.581 vs -0.777). v5 is marginally worse than even v4 (-0.748), placing it third out of three encoder variants.

### 2. Having BOTH high reward ρ AND high hand acc does NOT produce better policy

v5 was designed to have both axes simultaneously, and it does — yet it performs worse than v1, which had only weak reward ρ and near-chance hand accuracy. The combination of the two representation objectives did not translate to policy improvement.

### 3. No monotonic relationship between representation quality and policy performance

The ordering of representation quality metrics does not match the ordering of policy performance:

- Representation quality (combined): v4 ≈ v5 > v1
- Policy performance: v1 > v4 ≈ v5

This inversion suggests the representation axes (reward correlation, hand accuracy) measured during contrastive pre-training are not predictive of downstream policy utility in this REINFORCE setting.

---

## Hypotheses for Why v5 Underperforms v1

**1. Subspace partitioning fragments the useful signal.**
v1's 8-dim embedding is trained jointly to correlate with reward; even weakly, all 8 dims carry reward signal. v5 dedicates only dims 0–3 to reward and dims 4–7 to hand identity. The policy head receiving 8 dims gets a denser reward-relevant signal from v1 than from v5's 4 reward dims + 4 hand dims.

**2. Hand identity is not directly useful for policy.**
Knowing the opponent's likely hand is a representation-level property, but REINFORCE optimizes for action reward directly. The hand subspace dims may add noise to the policy head rather than useful information, because the mapping from "opponent's likely hand" to "best action" is not linear and requires further learning that 20,000 episodes may not be sufficient to discover.

**3. SupCon geometry is hostile to linear policy readout.**
The SupCon loss clusters hand classes together in embedding space. This creates a clustered geometry for dims 4–7, which the linear first layer of the policy head may not be able to leverage efficiently for action discrimination.

---

## Conclusion

The v5 frozen encoder is **not a better policy encoder** than v1. The hypothesis that "better representation quality (higher ρ + higher hand acc) → better policy performance" is **falsified** by this experiment.

The result suggests that contrastive pre-training objectives (reward ρ, opponent hand accuracy) are not sufficient proxies for policy-usefulness of a frozen representation. The v1 encoder, despite its weak reward correlation, may have incidentally captured low-dimensional structure that linearly predicts useful action-selection statistics.

**Best policy encoder so far**: `contrastive_repr_v1` (frozen L1 encoder), **-0.581 avg chips vs heuristic**.

---

## Output Files

- Checkpoint: `outputs/repr_policy_v1/run_frozen_v5encoder/checkpoint.pt`
- Training history: `outputs/repr_policy_v1/run_frozen_v5encoder/train_history.json`
- Eval result: `outputs/repr_policy_v1/run_frozen_v5encoder/eval_result.json`
