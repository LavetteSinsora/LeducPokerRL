"""Training entrypoint for dual-axis contrastive representation learning.

Usage:
    python -m experiments.representation_learning.dual_axis_repr_v1.train --smoke
    python -m experiments.representation_learning.dual_axis_repr_v1.train --episodes 20000 --output-dir outputs/dual_axis_repr_v1/run_default
"""

import argparse
import json
import os
from pathlib import Path
import time

import numpy as np
import torch

from experiments.representation_learning.dual_axis_repr_v1.agent import DualAxisReprAgent
from experiments.representation_learning.dual_axis_repr_v1.trainer import DualAxisReprTrainer


def run_pca_analysis(embeddings: np.ndarray):
    """Compute PCA on embeddings; return variance explained and effective dims.

    Args:
        embeddings: (N, D) array

    Returns:
        dict with 'variance_explained', 'effective_dim_80', 'effective_dim_90',
        'cumulative_variance'
    """
    from sklearn.decomposition import PCA

    n_components = min(embeddings.shape[0], embeddings.shape[1])
    pca = PCA(n_components=n_components)
    pca.fit(embeddings)

    var_explained = pca.explained_variance_ratio_.tolist()
    cumulative = np.cumsum(pca.explained_variance_ratio_)

    effective_dim_80 = int(np.searchsorted(cumulative, 0.80)) + 1
    effective_dim_80 = min(effective_dim_80, n_components)

    effective_dim_90 = int(np.searchsorted(cumulative, 0.90)) + 1
    effective_dim_90 = min(effective_dim_90, n_components)

    return {
        'variance_explained': var_explained,
        'cumulative_variance': cumulative.tolist(),
        'effective_dim_80': effective_dim_80,
        'effective_dim_90': effective_dim_90,
    }


def run_reward_spearman(embeddings: np.ndarray, rewards: np.ndarray):
    """Compute Spearman correlation between pairwise embedding distance and |ΔR|.

    Args:
        embeddings: (N, D) array
        rewards: (N,) array

    Returns:
        dict with 'spearman_rho' and 'spearman_pvalue'
    """
    from scipy.stats import spearmanr

    N = embeddings.shape[0]
    if N < 10:
        return {'spearman_rho': 0.0, 'spearman_pvalue': 1.0}

    max_pairs = 5000
    rng = np.random.default_rng(42)
    idx_i = rng.integers(0, N, size=max_pairs)
    idx_j = rng.integers(0, N, size=max_pairs)
    valid = idx_i != idx_j
    idx_i = idx_i[valid]
    idx_j = idx_j[valid]

    if len(idx_i) < 10:
        return {'spearman_rho': 0.0, 'spearman_pvalue': 1.0}

    diff = embeddings[idx_i] - embeddings[idx_j]
    embed_dists = np.linalg.norm(diff, axis=1)
    reward_diffs = np.abs(rewards[idx_i] - rewards[idx_j])

    rho, pval = spearmanr(embed_dists, reward_diffs)
    return {
        'spearman_rho': float(rho),
        'spearman_pvalue': float(pval),
    }


def run_hand_spearman(embeddings: np.ndarray, hand_labels: np.ndarray):
    """Compute Spearman correlation between pairwise embedding distance and hand label distance.

    Args:
        embeddings: (N, D) array
        hand_labels: (N,) integer array (0=J, 1=Q, 2=K)

    Returns:
        dict with 'spearman_rho' and 'spearman_pvalue'
    """
    from scipy.stats import spearmanr

    N = embeddings.shape[0]
    if N < 10:
        return {'spearman_rho': 0.0, 'spearman_pvalue': 1.0}

    max_pairs = 5000
    rng = np.random.default_rng(43)
    idx_i = rng.integers(0, N, size=max_pairs)
    idx_j = rng.integers(0, N, size=max_pairs)
    valid = idx_i != idx_j
    idx_i = idx_i[valid]
    idx_j = idx_j[valid]

    if len(idx_i) < 10:
        return {'spearman_rho': 0.0, 'spearman_pvalue': 1.0}

    diff = embeddings[idx_i] - embeddings[idx_j]
    embed_dists = np.linalg.norm(diff, axis=1)
    label_dists = np.abs(
        hand_labels[idx_i].astype(float) - hand_labels[idx_j].astype(float)
    )

    rho, pval = spearmanr(embed_dists, label_dists)
    return {
        'spearman_rho': float(rho),
        'spearman_pvalue': float(pval),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Train dual-axis contrastive encoder for Leduc Hold'em")

    # Loss hyperparameters
    parser.add_argument('--temperature', type=float, default=0.5,
                        help='InfoNCE temperature')
    parser.add_argument('--reward-thresh', type=float, default=0.5,
                        help='|ΔR| < this = reward-similar (positive pair condition)')
    parser.add_argument('--reward-margin', type=float, default=1.5,
                        help='|ΔR| > this = reward-dissimilar (negative pair condition)')
    parser.add_argument('--lambda-var', type=float, default=0.1,
                        help='VICReg variance weight')

    # Training settings
    parser.add_argument('--episodes', type=int, default=20000,
                        help='Total episodes to collect')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--batch-size', type=int, default=256,
                        help='Samples from replay buffer per update')
    parser.add_argument('--buffer-capacity', type=int, default=5000,
                        help='Replay buffer capacity')
    parser.add_argument('--episodes-per-step', type=int, default=8,
                        help='Episodes collected per training step')
    parser.add_argument('--embedding-dim', type=int, default=8,
                        help='Embedding dimensionality')

    # Data collection
    parser.add_argument('--data-agent-path', type=str,
                        default='agents/value_based/checkpoint.pt',
                        help='Path to trained value-based agent checkpoint')

    # Output
    parser.add_argument('--output-dir', type=str,
                        default=str(Path(__file__).parent / 'outputs' / 'run_default'),
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

    save_path = os.path.join(args.output_dir, 'checkpoint.pt')

    print(f"=== dual_axis_repr_v1 ===")
    print(f"Episodes: {args.episodes}, Batch: {args.batch_size}, "
          f"Buffer: {args.buffer_capacity}")
    print(f"Temperature: {args.temperature}, "
          f"reward_thresh: {args.reward_thresh}, "
          f"reward_margin: {args.reward_margin}")
    print(f"lambda_var: {args.lambda_var}, LR: {args.lr}")
    print()

    # Build agent
    agent = DualAxisReprAgent(embedding_dim=args.embedding_dim)

    # Build trainer
    trainer = DualAxisReprTrainer(
        agent=agent,
        data_agent_path=args.data_agent_path,
        learning_rate=args.lr,
        batch_size=args.batch_size,
        buffer_capacity=args.buffer_capacity,
        temperature=args.temperature,
        reward_thresh=args.reward_thresh,
        reward_margin=args.reward_margin,
        lambda_var=args.lambda_var,
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

    # Save loss log
    log_path = os.path.join(args.output_dir, 'loss_log.json')
    with open(log_path, 'w') as f:
        json.dump(loss_log, f)
    print(f"Loss log saved to {log_path}")

    # ------------------------------------------------------------------
    # Post-training diagnostics
    # ------------------------------------------------------------------
    print("\n=== Running diagnostics ===")
    agent.set_train_mode(False)

    all_states, all_rewards, all_labels = trainer.replay_buffer.get_all()
    print(f"Buffer size: {len(all_states)} samples")

    if len(all_states) < 20:
        print("Buffer too small for diagnostics.")
        results = {'error': 'buffer too small'}
    else:
        with torch.no_grad():
            all_embeddings = agent.encoder(all_states).numpy()
        rewards_np = all_rewards.numpy()
        labels_np = all_labels.numpy()

        # PCA
        print("Running PCA...")
        pca_results = run_pca_analysis(all_embeddings)
        print(f"  Variance per component: "
              f"{[f'{v:.3f}' for v in pca_results['variance_explained']]}")
        print(f"  Effective dimension (80% variance): {pca_results['effective_dim_80']}")
        print(f"  Effective dimension (90% variance): {pca_results['effective_dim_90']}")

        # Linear probe (opponent hand)
        print("Running linear probe (opponent hand)...")
        probe_accuracy = trainer.run_linear_probe()
        print(f"  Linear probe accuracy: {probe_accuracy:.3f} (chance = 0.333)")

        # Reward Spearman
        print("Running reward Spearman correlation...")
        reward_spearman = run_reward_spearman(all_embeddings, rewards_np)
        print(f"  Reward Spearman rho: {reward_spearman['spearman_rho']:.4f}, "
              f"p-value: {reward_spearman['spearman_pvalue']:.4g}")

        # Hand Spearman
        print("Running hand label Spearman correlation...")
        hand_spearman = run_hand_spearman(all_embeddings, labels_np)
        print(f"  Hand Spearman rho: {hand_spearman['spearman_rho']:.4f}, "
              f"p-value: {hand_spearman['spearman_pvalue']:.4g}")

        # Loss summary
        final_losses = (trainer.loss_history[-20:] if len(trainer.loss_history) >= 20
                        else trainer.loss_history)
        final_c_losses = (trainer.contrastive_loss_history[-20:]
                          if len(trainer.contrastive_loss_history) >= 20
                          else trainer.contrastive_loss_history)
        final_v_losses = (trainer.variance_loss_history[-20:]
                          if len(trainer.variance_loss_history) >= 20
                          else trainer.variance_loss_history)

        final_loss = float(np.mean(final_losses)) if final_losses else 0.0
        final_c_loss = float(np.mean(final_c_losses)) if final_c_losses else 0.0
        final_v_loss = float(np.mean(final_v_losses)) if final_v_losses else 0.0

        # Label distribution
        unique, counts = np.unique(labels_np, return_counts=True)
        label_dist = {int(k): int(v) for k, v in zip(unique, counts)}

        results = {
            'training': {
                'total_episodes': args.episodes,
                'temperature': args.temperature,
                'reward_thresh': args.reward_thresh,
                'reward_margin': args.reward_margin,
                'lambda_var': args.lambda_var,
                'lr': args.lr,
                'elapsed_seconds': round(elapsed, 1),
                'final_loss': round(final_loss, 6),
                'final_contrastive_loss': round(final_c_loss, 6),
                'final_variance_loss': round(final_v_loss, 6),
                'num_loss_steps': len(trainer.loss_history),
            },
            'linear_probe': {
                'accuracy': round(probe_accuracy, 4),
                'chance_baseline': 0.3333,
                'improvement_over_chance': round(probe_accuracy - 0.3333, 4),
            },
            'pca': pca_results,
            'reward_spearman': reward_spearman,
            'hand_spearman': hand_spearman,
            'buffer': {
                'size': len(all_states),
                'label_distribution': label_dist,
            },
        }

        print(f"\n=== dual_axis_repr_v1 Results ===")
        print(f"Effective dimension (80%): {pca_results['effective_dim_80']}")
        print(f"Linear probe accuracy (opp hand): {probe_accuracy:.3f} (chance=0.333)")
        print(f"Reward Spearman rho: {reward_spearman['spearman_rho']:.3f}")
        print(f"=================================")

    # Save results
    results_path = os.path.join(args.output_dir, 'results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")

    return results


if __name__ == '__main__':
    main()
