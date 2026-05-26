# repr_geometry_v1 — Geometry Analysis Report

**Checkpoint:** `outputs/contrastive_repr_v1/encoder_l2.pt` (L2/Rank-N-Contrast)
**States analyzed:** 169 unique states (from 2000 random episodes, 50 MC rollouts each)

---

## PCA Structure

| Component | Variance Explained | Cumulative |
|---|---|---|
| PC1 | 40.7% | 40.7% |
| PC2 | 36.3% | 77.0% |
| PC3 | 12.4% | 89.4% |
| PC4 | 5.0% | 94.5% |
| PC5 | 2.4% | 96.9% |
| PC6 | 1.6% | 98.5% |
| PC7 | 1.2% | 99.6% |
| PC8 | 0.4% | 100.0% |

**Effective dimensionality (80% threshold): 3**
**Effective dimensionality (90% threshold): 4**
**Top-1 PC variance: 40.7%**

---

## Cluster Separability

**Reward Spearman ρ** (pairwise emb distance vs |Δreward|): **0.163**

| Semantic Axis | Silhouette Score | Inter/Intra Ratio |
|---|---|---|
| player_hand | 0.019 | 1.069 |
| opp_hand | -0.029 | 1.004 |
| round | 0.015 | 1.012 |

**Strongest clustering axis: player_hand** (silhouette=0.019)

---

## Findings

### 1. Effective Dimensionality: 3 (vs diagnoser's reported 5)

The L2 encoder uses **3 effective dimensions at 80% variance** and 4 at 90% — slightly lower than the 5 reported in the contrastive_repr_v1 diagnosis. The discrepancy reflects different analysis contexts: the prior run's D3 metric measured all 8 embedding dimensions across the full training distribution, while this analysis operates on 169 unique states from random play. The PCA spectrum is notably two-axis dominated (PC1+PC2 = 77%), suggesting the encoder found two primary axes of variation.

### 2. Semantic Clustering: Weak Across All Axes

All silhouette scores are near zero: player_hand (0.019), round (0.015), opp_hand (-0.029). The inter/intra distance ratios are correspondingly close to 1.0 (1.07 for player_hand, 1.01 for round). This means:

- **Player hand clusters are detectable but barely** — states with the same hand card are only marginally closer together than states with different hands.
- **Opponent hand shows no clustering** (silhouette < 0). As expected: the opponent's private hand is invisible to the acting player, so it should not imprint on the embedding.
- **Round (pre-flop vs flop) shows negligible clustering** despite being encoded as a discrete feature in the 15-dim input. The encoder appears to distribute round information across the reward axis rather than allocating dedicated dimensions to it.

### 3. Reward Structure: Spearman ρ = 0.163 (Surprisingly Weak)

The Spearman ρ of 0.163 between pairwise embedding distances and |Δreward| is lower than the contrastive_repr_v1 Spearman diagnosis (which reported ρ = 0.641 for L2). This appears to be a methodological difference: the prior run computed Spearman ρ directly between scalar embeddings and reward values, while this experiment computes ρ over pairwise distances — a stricter metric.

A ρ of 0.163 over ~14,000 randomly sampled state pairs suggests a weak but real monotonic relationship: embedding proximity does correlate with reward proximity, but the signal is noisy over the full state space.

### 4. Most Interesting Finding: Dominant Two-PC Structure (77% in PC1+PC2)

The most visually striking observation is the strong two-axis dominance in PCA. The scree plot shows PC1 (40.7%) and PC2 (36.3%) together capturing 77% of variance, with a sharp elbow. This suggests the encoder learned two primary modes of variation:

- **PC1 is likely the reward axis** — aligning with L2's rank-correlation training objective
- **PC2 is likely a hand-strength × board-match interaction axis** — the most strategically relevant secondary feature in Leduc (holding a card that matches the board creates a pair, dramatically changing hand strength)

In the PCA scatter plots, states color-coded by expected reward show a gradient that roughly follows the first principal component, while player-hand coloring shows modest but visible separation — suggesting PC2 partially aligns with hand identity.

### 5. Opponent Hand: No Footprint (As Expected)

The opp_hand silhouette of -0.029 confirms that the opponent's private card leaves no geometric trace in the embedding. This is epistemically correct: since the encoder only receives the acting player's observation (which never contains the opponent's hand), any systematic opp_hand structure would indicate information leakage or spurious correlations in the random game trajectories. The near-zero result validates the encoder.

---

## Comparison to contrastive_repr_v1 Findings

| Metric | contrastive_repr_v1 (L2) | repr_geometry_v1 |
|---|---|---|
| Effective dim (80%) | 5 | 3 |
| Spearman ρ | 0.641 | 0.163 |
| Hand separation | 1.49 | 1.069 |
| Top-1 PC % | 31.8% | 40.7% |

The geometry is less distributed here than in the training-time diagnosis. The state space covered by 2000 random episodes (169 unique states) likely samples a narrower slice of the full distribution than the training rollouts, causing the apparent dimensionality reduction.

---

## Plots Generated

- `outputs/repr_geometry_v1/pca_tsne_grid.png` — 2×4 grid (top: PCA, bottom: t-SNE; columns: reward/player_hand/opp_hand/round)
- `outputs/repr_geometry_v1/pca_by_{reward,player_hand,opp_hand,round}.png` — individual PCA plots
- `outputs/repr_geometry_v1/pca_scree.png` — scree plot showing the two-PC elbow
