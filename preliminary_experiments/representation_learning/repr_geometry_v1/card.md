# repr_geometry_v1 — Geometry of the Reward-Contrastive Embedding Space

## Research Question

What is the geometric structure of the 8-dimensional embedding space learned by `contrastive_repr_v1`? Do strategically similar states cluster together?

## Motivation

`contrastive_repr_v1` trains an encoder (15 → 64 → 64 → 8) using terminal reward distance as the contrastive supervision signal. The training loss drives states with similar terminal rewards to have similar embeddings. But does this reward-proximity objective incidentally organize the space along other strategic axes?

Understanding the geometry reveals:
- Whether the encoder learned a rich representation or collapsed to a single axis
- Which semantic features (hand, board, round, reward) drive cluster structure
- Whether the embedding is suitable for downstream policy learning

## Key Questions

1. **Do states cluster by expected reward?** The training objective explicitly supervises this — it should be the dominant axis. How strongly?

2. **Do states cluster by player hand (J/Q/K)?** The player's private card is a key strategic feature. Does it emerge as a secondary axis even without explicit supervision?

3. **Do states cluster by opponent hand?** The opponent hand is hidden from the player — does it still leave a footprint in the embedding (via correlated game histories)?

4. **Do states cluster by game round?** Pre-flop vs flop states have fundamentally different information sets — does the encoder automatically segregate them?

## Checkpoint Used

- `outputs/contrastive_repr_v1/encoder_l2.pt` (L2 / Rank-N-Contrast formulation)
- L2 was chosen because it achieved the most distributed representation (5 effective dimensions) while retaining rich state identity information per the contrastive_repr_v1 report.

## Method

- Generate ~500–1000 unique states via 2000 random game episodes
- Compute expected reward via 50-rollout Monte Carlo for each state
- Encode all states with frozen encoder (eval mode, no_grad)
- Dimensionality reduction: PCA (all 8 components) + t-SNE (2D)
- Coloring: expected reward, player hand, opponent hand, game round
- Cluster metrics: silhouette score, intra/inter-cluster distance ratios, Spearman correlation
