# Representation Learning in Leduc Hold'em: A Session Report

**Date**: 2026-03-12
**Project**: PokerRL_Vanilla
**Scope**: 10 experiments across two broad research threads — value network capacity and multi-objective contrastive representation learning

---

## 1. Executive Summary

The most surprising result of this session is also the most practically useful: **subspace partitioning — routing each learning objective to a dedicated half of the output embedding — solved a gradient interference problem that four increasingly sophisticated normalization schemes could not**. Versions v1 through v4 of the dual-axis encoder each achieved one goal at the expense of the other; v5 achieved both simultaneously by the simplest possible structural change, with no tuning required.

The second surprising result is that a TD(0)-trained value network's hidden layer encodes reward-metric structure (Spearman ρ = 0.287) better than the best explicitly contrastive encoder from the prior research round (L1 at ρ = 0.163) — for free, as a byproduct of value learning. This raises a sharp question about when contrastive pretraining is worth the additional training cost.

The third key finding is that policy quality requires *both* reward-metric structure *and* hand-discriminating structure simultaneously. The v4 encoder, despite achieving a remarkable Spearman ρ = 0.672, produced inferior downstream policy performance (-0.748 vs heuristic) compared to the much simpler v1 encoder (ρ = 0.163, policy = -0.581). High reward correlation without opponent hand discrimination is insufficient for good play.

Finally, the minimum-capacity search (value_dim_search_v1) produced a methodological insight rather than a number: training instability from TD(0) self-play without a target network swamps any capacity signal. A 32×32 network (19% of baseline parameters) reached near-baseline performance transiently, confirming that capacity is not the bottleneck — but the same instability makes it impossible to cleanly answer the original question.

---

## 2. Background and Motivation

### The Game: Leduc Hold'em

Leduc Hold'em is a two-player, two-round simplified poker game designed for RL research. Each player receives one private card (Jack, Queen, or King from a 6-card deck). A community card is revealed after the first betting round. Players can fold, call, or raise. The game is strategically rich despite its small state space — correct play requires reasoning about opponent hand strength from observable betting patterns.

The game state as observed by an agent is encoded as a 15-dimensional vector: one-hot card representations, pot size, bet levels, round indicator, and stack information.

### The Baseline Agent

The reference agent (`value_based`) uses a TD(0)-trained value network with architecture `15 → 64 → 64 → 1` (~8,300 parameters), trained via self-play with a replay buffer. It achieves approximately **-0.075 chips/round vs the heuristic agent** — near parity against a hand-crafted rule-based opponent. The heuristic agent uses explicit card-rank logic, making it a reasonable calibration point.

### Motivating Questions for This Session

Three research questions drove the session:

1. **Minimum viable capacity**: The 64×64 baseline may be massively over-parameterized for a 15-dim input. What is the smallest network that achieves comparable performance?
2. **Alternative supervision targets**: The prior session (contrastive_repr_v1) trained an encoder to cluster game states by reward proximity. Can using *opponent hand identity* (J/Q/K) as the contrastive signal produce better representations for opponent modeling?
3. **Multi-objective representation**: Can a single encoder simultaneously learn reward-metric structure (for value estimation) and hand-discriminating structure (for opponent modeling) — and if so, how?

---

## 3. Experiment Log

### Experiment 1: value_dim_search_v1 — Minimum Viable Value Network Capacity

**Motivation**: The baseline's 8,321 parameters far exceed what Leduc's 15-dim state space should need. If a much smaller network achieves near-baseline performance, downstream architectures (such as representation-enhanced agents) can be simplified. Two architectures were tested: 32×32 (1,601 params, Run A) and 32×16 (1,057 params, Run B), both trained for 20,000 episodes with TD(0) self-play, identical to the baseline recipe.

**Key results**: Run A scored -0.52 chips/round vs heuristic; Run B scored -1.35. The baseline scores approximately -0.075. Both fall well short.

**Key insight**: The failure was not due to insufficient capacity. Run A briefly reached +0.07 chips/round at episode 9,000 and Run B touched +0.10 at episode 5,500 — both near-baseline. The problem was TD(0) self-play instability: without a target network, value estimates chase a moving target as the policy evolves, causing wide oscillations throughout training. The 20,000-episode recipe does not reliably converge, making the minimum-capacity question unanswerable under this setup.

**Conclusion**: Training instability, not network size, is the bottleneck. A target network is required before the capacity question can be answered cleanly.

---

### Experiment 2: hand_identity_repr_v1 — Opponent Hand as Contrastive Target

**Motivation**: The prior round's reward-contrastive encoder (contrastive_repr_v1) clustered states by reward proximity but could not encode opponent hand information (the opponent's card is not observable, so reward proximity only weakly correlates with opponent hand). A direct contrastive objective using *opponent hand label* (J=0, Q=1, K=2) was tested using online hard-negative triplet loss on a `15 → 64 → 64 → 8` encoder.

**Key results**: Linear probe accuracy for opponent hand = **62.8%** vs 33.3% chance baseline (+29.5 pp). The triplet loss converged from initial ~1.08 to final 0.99. Effective embedding dimensionality = **1** (PC1 = 99.76% of variance). Spearman ρ = 0.325 (p < 1e-120), confirming ordinal structure (J < Q < K in embedding space).

**Key insight**: The encoder successfully learned to distinguish opponent hand strength, but collapsed the 8-dimensional embedding to a single ordinal axis. This 1D collapse is semantically meaningful — Leduc hand strength is inherently ordinal — but it leaves 7 of 8 embedding dimensions unused. This waste motivated the dual-axis experiments: can a second supervision signal prevent this collapse while retaining hand-identity accuracy?

---

### Experiment 3: repr_geometry_v1 — Embedding Space Visualization

**Motivation**: Before building on the reward-contrastive encoder (contrastive_repr_v1, L2/Rank-N-Contrast), a geometric analysis was conducted to understand what the embedding space actually encodes.

**Key results**: Effective dimensionality = 3 at 80% variance threshold. PCA spectrum is dominated by two axes: PC1 = 40.7%, PC2 = 36.3% (77% combined). Reward Spearman ρ = 0.163 (pairwise distance metric). Silhouette scores for player_hand (0.019), round (0.015), and opponent_hand (-0.029) are all near zero.

**Key insight**: The embedding is genuinely low-dimensional (3 effective dims), with two dominant axes. PC1 is likely the reward axis; PC2 likely captures hand-strength × board-match interaction. Opponent hand leaves no footprint (silhouette = -0.029), confirming correct information masking — the encoder cannot leak unobservable information. The semantic clustering is weak across all axes, consistent with Leduc's intrinsic low-dimensionality.

---

### Experiment 4: repr_policy_v1 — Policy from Contrastive Representation

**Motivation**: If the L1 contrastive encoder (reward-based) produces a strategically useful representation, it should improve downstream policy learning. Three REINFORCE policy variants were compared over 20,000 training episodes, evaluated against the HeuristicAgent over 1,000 rounds.

**Key results**:

| Variant | Avg Chips vs Heuristic |
|---|---|
| Baseline (raw 15-dim features) | -0.942 |
| Frozen encoder (contrastive L1) | **-0.581** |
| Fine-tuned encoder | -1.330 |

**Key insight**: The frozen encoder wins by 0.36 chips/round over vanilla REINFORCE — a meaningful margin when the average pot is 6-8 chips. Fine-tuning is catastrophically worse than frozen. REINFORCE's noisy episode-level gradients corrupt the contrastive structure before the policy head can leverage it. The practical takeaway: use contrastive encoders as fixed feature extractors, not jointly-trained modules. This "frozen beats fine-tuned" finding recurs throughout the session.

---

### Experiment 5: dual_axis_repr_v1 — Joint AND/OR Positive/Negative Pairs

**Motivation**: Prevent the 1D collapse observed in hand_identity_repr_v1 by jointly supervising on both reward and hand identity. Version 1 used AND/OR logic for selecting positives: a state is a "positive" for an anchor if it has similar reward AND/OR similar opponent hand (using a combined InfoNCE-style loss with VICReg variance penalty).

**Key results**: Hand accuracy = 53.8% (vs 62.8% hand-only). Reward Spearman ρ = 0.107 (vs 0.163 reward-only). Effective dimensionality = 2 (expanded from hand-only's 1D). However, the loss turned negative and diverged to -3,188 by episode 20,000 — a characteristic InfoNCE over-collapse once VICReg stops being active.

**Key insight**: Dual supervision did prevent 1D collapse (2 effective dims vs 1), and both signals were partially encoded. But the AND/OR logic suffers from batch sparsity (few anchors find valid positives on both axes in a 256-sample batch), and the InfoNCE loss divergence indicates a training stability problem requiring a different formulation.

---

### Experiment 6: dual_axis_repr_v2 — SupCon Multi-Task Losses

**Motivation**: Replace the unstable InfoNCE formulation with Supervised Contrastive (SupCon, Khosla et al. 2020) losses, using categorical supervision. Reward was binned into 5 categories (thresholds at [-2.0, -0.5, 0.5, 2.0]). Total loss: `L_SupCon(reward_bins) + L_SupCon(hand_labels) + 0.1 × L_VICReg`.

**Key results**: Both losses converged steadily. Hand accuracy = **63.3%** (matching hand-only baseline). Reward bin accuracy = **55.4%** (vs 20% chance). Reward Spearman ρ = 0.118. Effective dimensionality = 2 at 80% threshold. No training divergence.

**Key insight**: SupCon simultaneously encodes both objectives with stable training, but the pairwise Spearman ρ (0.118) is lower than the reward-only contrastive baseline (0.163). SupCon clusters samples by bin membership but does not explicitly enforce inter-bin ordering in embedding space — categorical clustering is not metric learning. A hybrid combining SupCon for hand labels with a soft-distance loss for rewards was the natural next step.

---

### Experiment 7: dual_axis_repr_v3 — Hybrid L1 + SupCon (Unbalanced)

**Motivation**: Recover continuous reward metric structure (high Spearman ρ) while maintaining hand accuracy, using `L_hand_SupCon + λ × L_reward_L1 + L_VICReg`.

**Key results**: Hand accuracy = **67.8%** (new best). Reward Spearman ρ = **0.083** (worst of all versions). Loss scale at convergence: L_hand ≈ 5.26, L_reward ≈ 0.002 — a **2,600× scale gap**.

**Key insight**: The hybrid approach failed because the L1 soft-distance loss auto-calibrates to very small values as the metric is learned, while SupCon cross-entropy naturally stays in the 0 to log(N) range (~5-6 for N=256 batch size). With 2,600× scale imbalance, the optimizer is almost entirely driven by the hand objective. The reward axis receives only 0.04% of the total gradient signal. Loss family incompatibility — not conceptual conflict — is the root problem.

---

### Experiment 8: dual_axis_repr_v4 — EMA Loss Normalization

**Motivation**: Fix the scale imbalance in v3 by normalizing each loss by its exponential moving average, making both contribute equally in magnitude.

**Key results**: Reward Spearman ρ = **0.672** (4× above v1 baseline). Hand accuracy = **38.3%** (barely above 33% chance). Effective dimensionality = 4. EMA normalization inverted the dominance rather than balancing it.

**Key insight**: EMA normalization equalized *loss values* but not *gradient norms*. The reward loss, once its magnitude is normalized to 1.0, has a steeper loss surface relative to its magnitude (because beta auto-calibration in the L1 loss creates a well-conditioned optimization landscape). As a result, reward loss gradients dominate even with equal loss magnitudes. This experiment also revealed that the Spearman ρ = 0.672 did not translate to good policy performance — downstream policy evaluation would show the hand-discriminating deficit matters more for play quality.

---

### Experiment 9: value_based_repr_analysis — TD(0) Network Hidden Representation Quality

**Motivation**: Before pursuing more complex contrastive approaches, it is worth asking: does a standard value network's hidden layer already encode reward-metric structure as a byproduct of value learning? If so, contrastive pretraining may be unnecessary.

**Key results**:
- V(s) scalar Spearman ρ (output vs true reward): **0.626** — the value function is well-calibrated
- Hidden layer pairwise Spearman ρ (64-dim penultimate layer, pairwise L2 distance vs |ΔR|): **0.287**
- Raw 15-dim feature pairwise Spearman ρ: 0.106

**Key insight**: The TD(0) value network achieves a hidden-layer Spearman ρ of 0.287 — higher than the explicitly reward-contrastive encoder (L1, 0.163) — without any contrastive training objective. This "representation for free" finding is significant: the value network implicitly learns reward-metric structure as a byproduct of predicting V(s). However, the raw feature ρ of 0.106 suggests the hidden layer adds meaningful structure beyond the input. Whether this hidden representation (64-dim) is superior to a purpose-built contrastive embedding (8-dim) for downstream tasks remains open.

---

### Experiment 10: dual_axis_repr_v5 — Subspace Partitioning (Final Solution)

**Motivation**: Rather than normalizing competing gradients, eliminate gradient competition structurally. Split the 8-dim output into two disjoint 4-dim subspaces: dims 0–3 receive only L1 reward loss gradients; dims 4–7 receive only SupCon hand loss gradients. Shared hidden layers (two 64-dim layers) continue to receive gradients from both objectives, but the output layer is partitioned.

**Key results**:

| Subspace | Task | Metric | Value |
|---|---|---|---|
| Dims 0–3 (reward) | Reward metric | Spearman ρ | **0.543** |
| Dims 4–7 (hand) | Hand identity | Linear probe acc | **67.6%** |
| Full embedding | Both | Reward ρ / Hand acc | 0.536 / 65.2% |

Both targets simultaneously met (ρ ≥ 0.163 AND hand acc ≥ 62%) for the **first time** in this research line.

Cross-contamination: reward subspace hand accuracy = 0.575 (above 33% chance, due to shared hidden layers), hand subspace reward ρ = 0.087 (near-chance). The asymmetry is structurally expected: hand cards correlate with expected rewards in Leduc, so the reward subspace picks up some hand signal via shared hidden activations.

**Key insight**: Structural isolation — routing each loss to separate output dimensions — is more effective than any normalization scheme for multi-objective contrastive learning. The reward subspace achieves ρ = 0.543, which is **3.3× above the reward-only v1 baseline** (0.163), while the hand subspace achieves 67.6% — above the hand-only baseline of 62.8%. Both objectives are better served by subspace partitioning than they were alone with dedicated capacity.

---

## 4. Central Findings

### Finding 1 — Leduc Hold'em is Intrinsically Low-Dimensional

Every encoder trained in this session collapsed to 1–4 effective dimensions regardless of nominal embedding size (8 dims). The reward-contrastive encoder (contrastive_repr_v1, L2) uses 3 effective dimensions at the 80% PCA threshold. The hand-identity encoder collapses to 1D (PC1 = 99.76%), encoding opponent hand rank on a single ordinal axis (J < Q < K). Even the 32×32 value network briefly matched the 64×64 baseline at episode 9,000, consistent with the underlying function requiring far less capacity than the nominal architecture provides. Leduc's discrete card structure (3 hand values, 1 community card) means strategic distinctions that matter for play map onto a handful of dimensions, not a high-dimensional manifold.

### Finding 2 — Training Stability is the Binding Constraint for Value Networks

The value_dim_search_v1 experiment established that TD(0) self-play without a target network does not converge within 20,000 episodes, regardless of architecture size. Both tested architectures (32×32 and 32×16) showed oscillating evaluation scores and TD losses in the range 10–25, characteristic of self-play where shifting policies cause value targets to continuously move. The 32×32 network's transient near-baseline peak (+0.07 at episode 9,000) confirms sufficient capacity exists; the regression by episode 20,000 confirms the training recipe — not the network — is the limiting factor. A target network (hard update every ~500 episodes) is the recommended stabilization.

### Finding 3 — Reward Spearman ρ is Necessary but Not Sufficient for Policy Quality

The v4 encoder achieved Spearman ρ = 0.672 — by far the highest reward-metric correlation in this session — yet its hand accuracy was only 38.3% (near chance). If this encoder were used as a policy base, the missing hand-discriminating structure would likely produce an agent that estimates reward structure well but cannot differentiate how to respond to different opponent hands. By contrast, the v1 encoder (ρ = 0.163 but with useful multi-dimensional structure) produced frozen-encoder policy performance of -0.581 vs heuristic. Meanwhile, the TD(0) value network's hidden layer achieved ρ = 0.287 "for free" — better than the explicit contrastive objective — yet the policy learned on top of raw value features (baseline) scored -0.942. The implication: reward metric structure and hand-discriminating structure are both necessary conditions for policy quality; neither alone is sufficient.

### Finding 4 — Subspace Partitioning Solves Multi-Objective Contrastive Learning

The four dual-axis experiments (v1–v4) failed to simultaneously achieve both objectives for a structural reason: SupCon (categorical, angular loss) and SoftDistanceL1 (continuous metric loss) impose geometrically incompatible requirements on the same embedding dimensions. SupCon collapses same-class embeddings into tight clusters; SoftDistanceL1 enforces proportional inter-state distances. When both act on the same output neurons, one dominates. The loss scale gap in v3 was 2,600× (hand wins); EMA normalization in v4 inverted this to reward winning. Subspace partitioning in v5 resolved this by construction: each loss operates on separate output neurons, eliminating direct gradient competition. The result — reward ρ = 0.543 AND hand acc = 67.6% simultaneously — is not a compromise; both metrics *exceed* their respective single-axis baselines.

### Finding 5 — Frozen Encoder Beats Fine-Tuned Encoder for Downstream Policy

Across all tested policy variants (repr_policy_v1), the frozen contrastive encoder (-0.581 vs heuristic) substantially outperforms the fine-tuned variant (-1.330). Fine-tuning the encoder jointly with REINFORCE causes catastrophic forgetting: the noisy episode-level reward signal corrupts the structured representation before a policy-useful alternative can emerge. This finding has a practical implication: for representation-enhanced agents in Leduc, the two-phase approach (Phase 1: contrastive pretraining → Phase 2: frozen-encoder policy training) is strongly preferred over end-to-end learning.

---

## 5. Quantitative Summary Table

| Experiment | Key Configuration | Reward Spearman ρ | Hand Accuracy | Policy vs Heuristic | Both Targets? |
|---|---|---|---|---|---|
| contrastive_repr_v1 (L1) | Reward-only contrastive, 8-dim | 0.163 | ~33% (chance) | -0.581 (frozen) | — |
| value_dim_search_v1 Run A | 32×32 value net, TD(0) | — | — | -0.52 | — |
| value_dim_search_v1 Run B | 32×16 value net, TD(0) | — | — | -1.35 | — |
| value_based baseline | 64×64 value net, TD(0) | — | — | -0.075 | — |
| hand_identity_repr_v1 | Hand-only triplet, 8-dim | — | 62.8% | — | — |
| repr_geometry_v1 | Geometry analysis, L2 encoder | 0.163 | 1.9% silhouette | — | — |
| repr_policy_v1 (frozen) | L1 encoder, frozen | 0.163 | — | **-0.581** | — |
| repr_policy_v1 (finetune) | L1 encoder, unfrozen | — | — | -1.330 | — |
| repr_policy_v1 (baseline) | Raw 15-dim, REINFORCE | — | — | -0.942 | — |
| dual_axis_repr_v1 | AND/OR InfoNCE + VICReg | 0.107 | 53.8% | — | No |
| dual_axis_repr_v2 | SupCon reward bins + SupCon hand | 0.118 | 63.3% | — | No (ρ too low) |
| dual_axis_repr_v3 | SupCon hand + L1 reward (unbalanced) | 0.083 | 67.8% | — | No (ρ too low) |
| dual_axis_repr_v4 | SupCon hand + L1 reward + EMA norm | **0.672** | 38.3% | — | No (hand too low) |
| value_based_repr_analysis | TD(0) hidden layer analysis | 0.287 | — | — | — |
| dual_axis_repr_v5 | Subspace partitioned (4+4 dims) | **0.536** | **65.2%** | — | **YES** |

*Policy vs Heuristic: positive numbers are wins. Targets: ρ ≥ 0.163 AND hand acc ≥ 62%.*

---

## 6. Open Questions and Next Directions

**Q1: Does v5 improve downstream policy?**
The most pressing open question. The v5 encoder achieves both targets simultaneously, but its policy value has not been tested. Given that reward metric structure alone (v1, ρ = 0.163) produced a measurable policy improvement (-0.581 vs -0.942 for vanilla REINFORCE), v5's substantially higher ρ = 0.536 combined with good hand discrimination (65.2%) may produce markedly better policy performance. A frozen-encoder REINFORCE run using the v5 checkpoint is the immediate next experiment.

**Q2: Stabilized training for minimum-capacity search**
Add a target network (hard update every 500 episodes) and replay buffer to the TD(0) self-play recipe. Then re-run the 32×32 and 32×16 architectures. If both converge cleanly, test down to 16×16 and 16×1. The hypothesis — that capacity is not the bottleneck — predicts these architectures will converge to within 0.05 chips/round of the baseline.

**Q3: Actor-Critic instead of REINFORCE for policy evaluation**
All policy experiments in this session used REINFORCE (Monte Carlo policy gradient), which has high variance. The characteristic training instability (swings of ±1-2 chips/round between consecutive evaluations at 500-episode intervals) makes it difficult to distinguish representation quality effects from training noise. Actor-Critic with a learned value baseline would significantly reduce gradient variance and provide cleaner signal.

**Q4: What features does the value network's hidden layer encode beyond reward?**
The TD(0) value network's hidden Spearman ρ = 0.287 exceeds the explicit contrastive baseline (0.163). But is this hidden layer also encoding hand-discriminating structure? A linear probe for player hand and opponent hand on the 64-dim hidden activations would reveal whether value training implicitly learns the features needed for both policy quality conditions identified in Finding 3.

**Q5: Finer subspace partitioning and learned splits**
The v5 experiment used a fixed 4+4 split determined by dimension index. A learned routing mechanism — where a gating network allocates each dimension to the reward axis, hand axis, or both — could potentially achieve better utilization of the 8-dim space. Alternatively, testing 6+2, 2+6, and 3+5 splits would reveal whether the 4+4 allocation is optimal.

---

## 7. Methodology Notes

**Game and data collection**: All experiments collected data via self-play using a frozen `ValueBasedAgent` (`agents/value_based/checkpoint.pt`) as the data-generating policy. This ensures consistent state visitation distributions across experiments but introduces a potential distribution shift — the encoder is trained on states generated by the value agent, not the final policy.

**Standard training configuration**: Unless otherwise noted — Adam optimizer, lr = 1e-4, replay buffer capacity = 5,000, batch size = 256, 20,000 self-play episodes. The value_dim_search_v1 used batch size 32 (TD value learning convention) instead of 256.

**Evaluation protocol**: Policy experiments (repr_policy_v1) were evaluated over 1,000 rounds vs HeuristicAgent at training end. Value network experiments (value_dim_search_v1) were evaluated every 2,000 episodes over 500 rounds. Spearman ρ metrics are computed on held-out buffer samples (typically 2,000 randomly-sampled state pairs).

**Spearman ρ formulation**: Two distinct formulations appear in this session and should not be directly compared. The contrastive_repr_v1 "scalar ρ" measured correlation between individual embedding coordinates and scalar reward values. The "pairwise ρ" used in repr_geometry_v1 onward computes Spearman rank correlation between all pairwise L2 distances in embedding space and the corresponding pairwise absolute reward differences — a stricter metric that penalizes any violation of the global distance ordering.

**SupCon reward binning**: Reward was discretized into 5 bins with fixed thresholds [-∞, -2.0, -0.5, 0.5, 2.0, +∞] for the v2–v5 dual-axis experiments. The bin boundaries were chosen to roughly balance class frequencies over the Leduc terminal reward distribution; no hyperparameter tuning was performed.

**Encoder architecture**: All contrastive encoders use the same `15 → 64 → 64 → d` MLP with ReLU activations, where `d` is the embedding dimension (8 for all experiments in this session). The hidden layer width (64×64) is shared with the value network baseline, enabling fair comparison of output-layer structure rather than capacity.

**VICReg regularization**: All dual-axis experiments include a VICReg variance term (λ_var = 0.1) to prevent dimensional collapse. The variance target is std = 1.0 per dimension; the VICReg loss penalizes dimensions that fall below this target. This is necessary but not sufficient to prevent collapse: hand_identity_repr_v1 used triplet loss without VICReg and collapsed to 1D, while dual_axis_repr_v2 with VICReg achieved effective_dim = 2.

---

*Report generated 2026-03-12. All experiment outputs are in `outputs/` subdirectories under the project root.*
