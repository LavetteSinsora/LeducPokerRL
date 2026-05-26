"""
D3: Public State (P_t) Representational Quality

The StateEncoder maps game events to a sequence of P_t vectors (8-dim).
If P_t is informative, vectors at different game positions should be
distinguishable; vectors for the same position should cluster together.

Metrics:
  - Mean pairwise cosine distance between P_t at consecutive positions
    (step 0 → step 1 → ... step 5). Low = P_t barely changes per event.
  - Within-position variance: how consistent is P_t for the same step?
    High = the encoder is sensitive to game trajectory (good).
  - Norm of P_t over time: collapsing norms → dying representations.

Positions tracked:
  0: initial zero vector (game start)
  1: after P0's first preflop action
  2: after P1's first preflop action (or earlier if fold)
  3: after community card deal
  4: after first postflop action
  5: after second postflop action

Visualization:
  - d3_cosine_dist.png: step-to-step cosine distance heatmaps
  - d3_norm.png:        P_t vector norms per step over checkpoints
  - d3_tsne.png:        2D t-SNE of P_t vectors colored by event_index
  - d3_within_var.png:  within-position variance per checkpoint

Output:
  outputs/d3_*.png
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import defaultdict

try:
    from sklearn.manifold import TSNE
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

from diagnose import (
    CHECKPOINT_EPISODES, load_checkpoint, run_greedy_games,
    ensure_output_dir, OUTPUT_DIR, COLORS,
)

N_GAMES = 800


def cosine_distance(a, b):
    """Cosine distance = 1 - cosine_similarity."""
    norm_a = np.linalg.norm(a) + 1e-8
    norm_b = np.linalg.norm(b) + 1e-8
    return 1.0 - float(np.dot(a, b) / (norm_a * norm_b))


def analyze_pubstate(records):
    """
    Collect P_t vectors per event index across all games.
    Returns:
      by_step: dict[step_idx] → list of P_t vectors (np.ndarray d)
    """
    by_step = defaultdict(list)
    for rec in records:
        for step_i, P_t in enumerate(rec.P_at_events):
            by_step[step_i].append(np.array(P_t))
    return by_step


def step_to_step_distances(by_step, max_steps=7):
    """
    Compute mean cosine distance from step i → step i+1.
    Returns list of (step_i, step_j, mean_cosine_dist).
    """
    dists = []
    steps = sorted(s for s in by_step.keys() if s <= max_steps)
    for i in range(len(steps) - 1):
        s0, s1 = steps[i], steps[i + 1]
        vecs0 = by_step[s0]
        vecs1 = by_step[s1]
        # Pair up by game (same position in game)
        n = min(len(vecs0), len(vecs1))
        cd = [cosine_distance(vecs0[k], vecs1[k]) for k in range(n)]
        dists.append((s0, s1, float(np.mean(cd))))
    return dists


def within_position_variance(by_step, max_steps=7):
    """
    Mean L2 variance within each step's P_t vectors.
    High variance = encoder is sensitive to game trajectory.
    """
    variances = {}
    for step, vecs in by_step.items():
        if step > max_steps or len(vecs) < 2:
            continue
        arr = np.array(vecs)           # (N, d)
        var = float(arr.var(axis=0).mean())  # mean variance across dimensions
        variances[step] = var
    return variances


def mean_norms(by_step, max_steps=7):
    """Mean L2 norm of P_t at each step."""
    norms = {}
    for step, vecs in by_step.items():
        if step > max_steps:
            continue
        arr = np.array(vecs)
        norms[step] = float(np.linalg.norm(arr, axis=1).mean())
    return norms


def main():
    ensure_output_dir()

    ep_results = {}

    for ep in CHECKPOINT_EPISODES:
        print(f"D3: ep {ep:,} — playing {N_GAMES} games...")
        config, state_enc, belief_net, q_net = load_checkpoint(ep)
        records = run_greedy_games(state_enc, belief_net, q_net, config, n_games=N_GAMES)
        by_step = analyze_pubstate(records)
        dists   = step_to_step_distances(by_step)
        variances = within_position_variance(by_step)
        norms   = mean_norms(by_step)
        ep_results[ep] = {
            'by_step': by_step, 'dists': dists,
            'variances': variances, 'norms': norms, 'records': records,
        }
        avg_dist = np.mean([d for _, _, d in dists]) if dists else 0
        print(f"  avg step-to-step cosine dist: {avg_dist:.4f}  "
              f"max norm: {max(norms.values(), default=0):.4f}")

    ep_colors = dict(zip(CHECKPOINT_EPISODES, COLORS))

    # ── Plot 1: Step-to-step cosine distance per checkpoint ───────────────────
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_title('D3: Step-to-Step Cosine Distance in P_t\n'
                 '(higher = more change per event; near 0 = P_t barely moves)', fontsize=11)

    for ep in CHECKPOINT_EPISODES:
        dists = ep_results[ep]['dists']
        if not dists:
            continue
        xs = [f'{s0}→{s1}' for s0, s1, _ in dists]
        ys = [d for _, _, d in dists]
        ax.plot(range(len(xs)), ys, 'o-', color=ep_colors[ep], lw=2, ms=7, label=f'ep {ep:,}')
    ax.set_xticks(range(len(xs)))
    ax.set_xticklabels(xs, fontsize=9)
    ax.set_xlabel('Event Step Transition', fontsize=10)
    ax.set_ylabel('Mean Cosine Distance', fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    path = f'{OUTPUT_DIR}/d3_cosine_dist.png'
    plt.savefig(path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"\n  Saved: {path}")

    # ── Plot 2: P_t vector norms per step ─────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_title('D3: P_t Mean L2 Norm per Event Step\n'
                 '(collapsing norms → dying representation)', fontsize=11)

    for ep in CHECKPOINT_EPISODES:
        norms = ep_results[ep]['norms']
        steps = sorted(norms.keys())
        vals  = [norms[s] for s in steps]
        ax.plot(steps, vals, 'o-', color=ep_colors[ep], lw=2, ms=7, label=f'ep {ep:,}')
    ax.set_xlabel('Event Step Index', fontsize=10)
    ax.set_ylabel('Mean Norm of P_t', fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    path = f'{OUTPUT_DIR}/d3_norm.png'
    plt.savefig(path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")

    # ── Plot 3: Within-position variance per checkpoint ───────────────────────
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_title('D3: Within-Position Variance of P_t\n'
                 '(higher = encoder is sensitive to game trajectory; '
                 'near 0 = same output regardless of how we got here)', fontsize=11)

    for ep in CHECKPOINT_EPISODES:
        variances = ep_results[ep]['variances']
        steps = sorted(variances.keys())
        vals  = [variances[s] for s in steps]
        ax.plot(steps, vals, 'o-', color=ep_colors[ep], lw=2, ms=7, label=f'ep {ep:,}')
    ax.set_xlabel('Event Step Index', fontsize=10)
    ax.set_ylabel('Mean Dimension-Wise Variance', fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    path = f'{OUTPUT_DIR}/d3_within_var.png'
    plt.savefig(path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")

    # ── Plot 4: t-SNE of P_t (if sklearn available) ───────────────────────────
    if HAS_SKLEARN:
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('D3: t-SNE of P_t Vectors per Checkpoint\n'
                     '(colored by event step; clusters = same-step vectors group together)', fontsize=12)

        step_cmap = plt.cm.get_cmap('plasma', 8)

        for idx, ep in enumerate(CHECKPOINT_EPISODES):
            ax = axes[idx // 3][idx % 3]
            by_step = ep_results[ep]['by_step']

            # Collect up to 100 vectors per step
            all_vecs = []
            all_labels = []
            for step, vecs in sorted(by_step.items()):
                if step > 6:
                    continue
                sample = vecs[:100]
                all_vecs.extend(sample)
                all_labels.extend([step] * len(sample))

            if len(all_vecs) < 10:
                ax.text(0.5, 0.5, 'Not enough data', ha='center', va='center')
                continue

            arr = np.array(all_vecs)
            perplexity = min(30, len(arr) - 1)
            tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42, n_iter=500)
            xy = tsne.fit_transform(arr)

            scatter = ax.scatter(xy[:, 0], xy[:, 1],
                                 c=all_labels, cmap='plasma', vmin=0, vmax=7,
                                 s=12, alpha=0.6)
            plt.colorbar(scatter, ax=ax, label='event step')
            ax.set_title(f'ep {ep:,}', fontsize=10)
            ax.set_xlabel('t-SNE 1', fontsize=8)
            ax.set_ylabel('t-SNE 2', fontsize=8)

        plt.tight_layout()
        path = f'{OUTPUT_DIR}/d3_tsne.png'
        plt.savefig(path, dpi=100, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {path}")
    else:
        print("  (sklearn not available — skipping t-SNE plot)")

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n── D3 Summary: avg cosine distance between consecutive steps ─────────")
    for ep in CHECKPOINT_EPISODES:
        dists = ep_results[ep]['dists']
        avg = np.mean([d for _, _, d in dists]) if dists else 0
        max_norm = max(ep_results[ep]['norms'].values(), default=0)
        print(f"  ep {ep:>7,}: avg_step_dist={avg:.4f}  max_norm={max_norm:.4f}")


if __name__ == '__main__':
    main()
