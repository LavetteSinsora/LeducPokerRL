"""Training entrypoint for hand-identity representation learning.

Usage:
    python -m experiments.representation_learning.hand_identity_repr_v1.train --loss-type triplet
    python -m experiments.representation_learning.hand_identity_repr_v1.train --loss-type ce
    python -m experiments.representation_learning.hand_identity_repr_v1.train --smoke
    python -m experiments.representation_learning.hand_identity_repr_v1.train --episodes 20000 --output-dir outputs/hand_identity_repr_v1/run_triplet
"""

import argparse
import json
import os
from pathlib import Path
import time

import numpy as np
import torch

from experiments.representation_learning.hand_identity_repr_v1.agent import HandIdentityReprAgent
from experiments.representation_learning.hand_identity_repr_v1.trainer import HandIdentityReprTrainer


def run_pca_analysis(embeddings: np.ndarray):
    """Compute PCA on embeddings, return variance explained per component and effective_dim_80.

    Args:
        embeddings: (N, D) array

    Returns:
        dict with 'variance_explained', 'effective_dim_80', 'cumulative_variance'
    """
    from sklearn.decomposition import PCA

    n_components = min(embeddings.shape[0], embeddings.shape[1])
    pca = PCA(n_components=n_components)
    pca.fit(embeddings)

    var_explained = pca.explained_variance_ratio_.tolist()
    cumulative = np.cumsum(pca.explained_variance_ratio_)

    # How many components for 80% cumulative variance?
    effective_dim_80 = int(np.searchsorted(cumulative, 0.80)) + 1
    effective_dim_80 = min(effective_dim_80, n_components)

    return {
        'variance_explained': var_explained,
        'cumulative_variance': cumulative.tolist(),
        'effective_dim_80': effective_dim_80,
    }


def run_spearman_analysis(embeddings: np.ndarray, labels: np.ndarray):
    """Compute Spearman correlation between embedding distance and hand label distance.

    Args:
        embeddings: (N, D) array
        labels: (N,) integer array

    Returns:
        dict with 'spearman_rho' and 'spearman_pvalue'
    """
    from scipy.stats import spearmanr

    N = embeddings.shape[0]
    if N < 10:
        return {'spearman_rho': 0.0, 'spearman_pvalue': 1.0}

    # Subsample to avoid memory issues with large N
    max_pairs = 5000
    rng = np.random.default_rng(42)
    idx_i = rng.integers(0, N, size=max_pairs)
    idx_j = rng.integers(0, N, size=max_pairs)

    # Filter out self-pairs
    valid = idx_i != idx_j
    idx_i = idx_i[valid]
    idx_j = idx_j[valid]

    if len(idx_i) < 10:
        return {'spearman_rho': 0.0, 'spearman_pvalue': 1.0}

    # Embedding L2 distances
    diff = embeddings[idx_i] - embeddings[idx_j]
    embed_dists = np.linalg.norm(diff, axis=1)

    # Hand label distances (|label_i - label_j|)
    label_dists = np.abs(labels[idx_i].astype(float) - labels[idx_j].astype(float))

    rho, pval = spearmanr(embed_dists, label_dists)
    return {
        'spearman_rho': float(rho),
        'spearman_pvalue': float(pval),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Train hand-identity representation encoder for Leduc Hold'em")

    # Core settings
    parser.add_argument('--loss-type', type=str, default='triplet',
                        choices=['triplet', 'ce', 'both'],
                        help='Loss formulation: triplet, ce (cross-entropy), or both')
    parser.add_argument('--episodes', type=int, default=20000,
                        help='Total episodes to collect')
    parser.add_argument('--margin', type=float, default=1.0,
                        help='Triplet loss margin')
    parser.add_argument('--temperature', type=float, default=0.5,
                        help='Temperature (reserved for future loss variants)')
    parser.add_argument('--embedding-dim', type=int, default=8,
                        help='Embedding dimensionality')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--batch-size', type=int, default=256,
                        help='Samples from replay buffer per update')
    parser.add_argument('--buffer-capacity', type=int, default=5000,
                        help='Replay buffer capacity')
    parser.add_argument('--episodes-per-step', type=int, default=8,
                        help='Episodes collected per training step')

    # Data collection
    parser.add_argument('--data-agent-path', type=str,
                        default='agents/value_based/checkpoint.pt',
                        help='Path to trained value-based agent checkpoint')

    # Output
    parser.add_argument('--output-dir', type=str,
                        default=str(Path(__file__).parent / 'outputs'),
                        help='Directory for checkpoints and logs')
    parser.add_argument('--smoke', action='store_true',
                        help='Quick sanity check (200 episodes, small batch)')

    args = parser.parse_args()

    # Smoke test overrides
    if args.smoke:
        args.episodes = 200
        args.batch_size = 64
        args.buffer_capacity = 500
        args.episodes_per_step = 8

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    save_path = os.path.join(args.output_dir, f'encoder_{args.loss_type}.pt')

    print(f"=== Hand Identity Repr v1: {args.loss_type} ===")
    print(f"Episodes: {args.episodes}, Batch: {args.batch_size}, "
          f"Buffer: {args.buffer_capacity}")
    print(f"Margin: {args.margin}, LR: {args.lr}")
    print()

    # Build agent
    use_ce_head = args.loss_type in ('ce', 'both')
    agent = HandIdentityReprAgent(
        embedding_dim=args.embedding_dim,
        use_classification_head=use_ce_head,
    )

    # Build trainer
    trainer = HandIdentityReprTrainer(
        agent=agent,
        data_agent_path=args.data_agent_path,
        loss_type=args.loss_type,
        learning_rate=args.lr,
        batch_size=args.batch_size,
        buffer_capacity=args.buffer_capacity,
        margin=args.margin,
        episodes_per_step=args.episodes_per_step,
    )

    # Train
    t0 = time.time()
    loss_log = []

    def log_callback(info):
        if info.get('type') == 'batch_update':
            loss_log.append({
                'episode': info['episode'],
                'loss': info['loss'],
            })

    trainer.train(
        num_episodes=args.episodes,
        save_path=save_path,
        callback=log_callback,
    )

    elapsed = time.time() - t0
    print(f"\nTraining complete in {elapsed:.1f}s")
    print(f"Checkpoint saved to {save_path}")

    # ------------------------------------------------------------------
    # Post-training diagnostics
    # ------------------------------------------------------------------
    print("\n=== Running diagnostics ===")
    agent.set_train_mode(False)

    # Get all buffered states and labels
    all_states, all_labels = trainer.replay_buffer.get_all()
    print(f"Buffer size: {len(all_states)} samples")

    if len(all_states) < 20:
        print("Buffer too small for diagnostics.")
        results = {'error': 'buffer too small'}
    else:
        with torch.no_grad():
            all_embeddings = agent.encoder(all_states).numpy()
        labels_np = all_labels.numpy()

        # PCA analysis
        print("Running PCA...")
        pca_results = run_pca_analysis(all_embeddings)
        print(f"  Variance explained per component: "
              f"{[f'{v:.3f}' for v in pca_results['variance_explained']]}")
        print(f"  Effective dimension (80% variance): {pca_results['effective_dim_80']}")

        # Linear probe
        print("Running linear probe...")
        probe_accuracy = trainer.run_linear_probe()
        print(f"  Linear probe accuracy: {probe_accuracy:.3f} (chance = 0.333)")

        # Spearman correlation
        print("Running Spearman correlation...")
        spearman_results = run_spearman_analysis(all_embeddings, labels_np)
        print(f"  Spearman rho: {spearman_results['spearman_rho']:.4f}, "
              f"p-value: {spearman_results['spearman_pvalue']:.4g}")

        # Loss summary
        final_losses = [e['loss'] for e in loss_log[-20:]] if len(loss_log) >= 20 else \
                       [e['loss'] for e in loss_log]
        final_loss = float(np.mean(final_losses)) if final_losses else 0.0

        # Label distribution in buffer
        unique, counts = np.unique(labels_np, return_counts=True)
        label_dist = {int(k): int(v) for k, v in zip(unique, counts)}

        results = {
            'training': {
                'total_episodes': args.episodes,
                'loss_type': args.loss_type,
                'margin': args.margin,
                'lr': args.lr,
                'elapsed_seconds': round(elapsed, 1),
                'final_loss': round(final_loss, 6),
                'num_loss_steps': len(loss_log),
            },
            'linear_probe': {
                'accuracy': round(probe_accuracy, 4),
                'chance_baseline': 0.3333,
                'improvement_over_chance': round(probe_accuracy - 0.3333, 4),
            },
            'pca': pca_results,
            'spearman': spearman_results,
            'buffer': {
                'size': len(all_states),
                'label_distribution': label_dist,
            },
        }

        print("\n=== Summary ===")
        print(f"Final loss (last 20 steps avg): {final_loss:.6f}")
        print(f"Linear probe accuracy:          {probe_accuracy:.3f}  (chance = 0.333)")
        print(f"Effective dimension (80% var):  {pca_results['effective_dim_80']}")
        print(f"Spearman rho:                   {spearman_results['spearman_rho']:.4f}")

    # Save results
    results_path = os.path.join(args.output_dir, 'results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")

    # Save loss log
    log_path = os.path.join(args.output_dir, f'loss_log_{args.loss_type}.json')
    with open(log_path, 'w') as f:
        json.dump(loss_log, f)
    print(f"Loss log saved to {log_path}")

    return results


if __name__ == '__main__':
    main()
