"""
D4: Event Embedding Analysis

Extracts the 9×8 event embedding matrix from state_encoder.event_embed at
each checkpoint and asks: are embeddings semantically structured?

Expected structure if working well:
  - Action embeddings (0–5) should differ from deal embeddings (6–8)
  - Same-action-different-player (P0_raise vs P1_raise) may be similar
  - Deal embeddings for J/Q/K should differ (different board info)

If embeddings collapse (high pairwise similarity, near-zero PCA spread),
the state encoder cannot distinguish what just happened in the game.

Output:
  outputs/d4_similarity_{ep}.png  — cosine similarity heatmaps per checkpoint
  outputs/d4_pca_all.png          — PCA scatter of all 9 embeddings × 6 checkpoints
  outputs/d4_cluster_distance.png — action-vs-deal cluster distance over time
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import torch
import torch.nn.functional as F

from diagnose import (
    CHECKPOINT_EPISODES, load_checkpoint, ensure_output_dir, OUTPUT_DIR,
    EVENT_LABELS, COLORS,
)


def cosine_sim_matrix(W: np.ndarray) -> np.ndarray:
    """W shape (9, d). Returns (9, 9) cosine similarity matrix."""
    norms = np.linalg.norm(W, axis=1, keepdims=True) + 1e-8
    W_n = W / norms
    return W_n @ W_n.T


def pca_2d(W: np.ndarray) -> np.ndarray:
    """Simple 2D PCA. W shape (N, d). Returns (N, 2)."""
    W_c = W - W.mean(axis=0)
    U, S, Vt = np.linalg.svd(W_c, full_matrices=False)
    return W_c @ Vt[:2].T


def action_deal_distance(W: np.ndarray) -> float:
    """Mean pairwise L2 distance between action embeddings (0–5) and deal embeddings (6–8)."""
    actions = W[:6]
    deals = W[6:]
    dists = []
    for a in actions:
        for d in deals:
            dists.append(np.linalg.norm(a - d))
    return float(np.mean(dists))


def within_cluster_similarity(W: np.ndarray) -> float:
    """Mean cosine similarity within all 9 embeddings (off-diagonal). Higher = more collapsed."""
    sim = cosine_sim_matrix(W)
    n = sim.shape[0]
    off_diag = sim[~np.eye(n, dtype=bool)]
    return float(off_diag.mean())


def main():
    ensure_output_dir()

    all_weights = {}     # ep → np.ndarray (9, 8)
    all_distances = []   # action-deal distance per ep
    all_mean_sim = []    # mean off-diagonal similarity per ep

    print("D4: Loading checkpoints and extracting embeddings...")
    for ep in CHECKPOINT_EPISODES:
        _, state_enc, _, _ = load_checkpoint(ep)
        W = state_enc.event_embed.weight.detach().numpy().copy()  # (9, 8)
        all_weights[ep] = W
        all_distances.append(action_deal_distance(W))
        all_mean_sim.append(within_cluster_similarity(W))
        print(f"  ep {ep:>7d}: action-deal dist={all_distances[-1]:.4f}  mean_sim={all_mean_sim[-1]:.4f}")

    # ── Plot 1: Cosine similarity heatmaps ──────────────────────────────────
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('D4: Event Embedding Cosine Similarity Matrix\n(off-diagonal similarity → higher = more collapsed)', fontsize=13)

    for idx, ep in enumerate(CHECKPOINT_EPISODES):
        ax = axes[idx // 3][idx % 3]
        sim = cosine_sim_matrix(all_weights[ep])
        im = ax.imshow(sim, cmap='RdYlGn', vmin=-1, vmax=1)
        ax.set_xticks(range(9))
        ax.set_yticks(range(9))
        ax.set_xticklabels(EVENT_LABELS, rotation=45, ha='right', fontsize=7)
        ax.set_yticklabels(EVENT_LABELS, fontsize=7)
        ax.set_title(f'ep {ep:,}', fontsize=10)

        # Add value annotations
        for i in range(9):
            for j in range(9):
                ax.text(j, i, f'{sim[i, j]:.2f}', ha='center', va='center',
                        fontsize=5, color='black')

        plt.colorbar(im, ax=ax, shrink=0.8)

    plt.tight_layout()
    path = f'{OUTPUT_DIR}/d4_similarity_heatmaps.png'
    plt.savefig(path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")

    # ── Plot 2: PCA scatter — all checkpoints overlaid ───────────────────────
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('D4: Event Embeddings — 2D PCA\n(9 events per checkpoint; well-separated = structured)', fontsize=13)

    action_markers = ['s', 's', 's', 'o', 'o', 'o']      # P0=square, P1=circle
    deal_markers   = ['^', '^', '^']                        # deals=triangle
    all_markers    = action_markers + deal_markers
    action_colors  = ['#1976D2', '#388E3C', '#D32F2F',     # P0: fold/call/raise
                      '#42A5F5', '#66BB6A', '#EF5350']      # P1: fold/call/raise
    deal_colors    = ['#FF6F00', '#F57F17', '#BF360C']      # deal J/Q/K
    all_colors     = action_colors + deal_colors

    for idx, ep in enumerate(CHECKPOINT_EPISODES):
        ax = axes[idx // 3][idx % 3]
        W = all_weights[ep]
        xy = pca_2d(W)

        for i, label in enumerate(EVENT_LABELS):
            ax.scatter(xy[i, 0], xy[i, 1],
                       c=all_colors[i], marker=all_markers[i],
                       s=80, zorder=3)
            ax.annotate(label, (xy[i, 0], xy[i, 1]),
                        textcoords='offset points', xytext=(4, 4), fontsize=7)

        ax.axhline(0, color='gray', lw=0.5, ls='--')
        ax.axvline(0, color='gray', lw=0.5, ls='--')
        ax.set_title(f'ep {ep:,}', fontsize=10)
        ax.set_xlabel('PC1', fontsize=8)
        ax.set_ylabel('PC2', fontsize=8)

    plt.tight_layout()
    path = f'{OUTPUT_DIR}/d4_pca_all.png'
    plt.savefig(path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")

    # ── Plot 3: Action-deal distance and mean similarity over episodes ────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle('D4: Embedding Structure Over Training', fontsize=13)

    eps = [ep / 1000 for ep in CHECKPOINT_EPISODES]

    ax1.plot(eps, all_distances, 'o-', color='#1565C0', lw=2, ms=8)
    ax1.set_xlabel('Episode (thousands)', fontsize=10)
    ax1.set_ylabel('Mean Action–Deal L2 Distance', fontsize=10)
    ax1.set_title('Action vs Deal Cluster Separation\n(higher = better separated)', fontsize=10)
    ax1.grid(alpha=0.3)
    for x, y in zip(eps, all_distances):
        ax1.annotate(f'{y:.3f}', (x, y), textcoords='offset points', xytext=(0, 8), fontsize=8, ha='center')

    ax2.plot(eps, all_mean_sim, 'o-', color='#B71C1C', lw=2, ms=8)
    ax2.axhline(0, color='gray', ls='--', lw=1)
    ax2.set_xlabel('Episode (thousands)', fontsize=10)
    ax2.set_ylabel('Mean Off-Diagonal Cosine Similarity', fontsize=10)
    ax2.set_title('Embedding Collapse Metric\n(higher = more collapsed / less distinct)', fontsize=10)
    ax2.grid(alpha=0.3)
    for x, y in zip(eps, all_mean_sim):
        ax2.annotate(f'{y:.3f}', (x, y), textcoords='offset points', xytext=(0, 8), fontsize=8, ha='center')

    plt.tight_layout()
    path = f'{OUTPUT_DIR}/d4_structure_over_time.png'
    plt.savefig(path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")

    # ── Print summary ──────────────────────────────────────────────────────────
    print("\n── D4 Summary ──────────────────────────────────────────────────────")
    print(f"{'Episode':>10}  {'Action-Deal Dist':>18}  {'Mean Cosine Sim':>17}")
    for ep, dist, sim in zip(CHECKPOINT_EPISODES, all_distances, all_mean_sim):
        print(f"{ep:>10,}  {dist:>18.4f}  {sim:>17.4f}")
    print()


if __name__ == '__main__':
    main()
