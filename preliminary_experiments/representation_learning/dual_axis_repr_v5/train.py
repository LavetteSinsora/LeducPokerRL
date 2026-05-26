"""Training entrypoint for dual-axis subspace-partitioned representation learning (v5).

Structural gradient isolation: reward loss applied only to dims 0:4,
hand loss applied only to dims 4:8. No interference by construction.

Usage:
    python -m experiments.representation_learning.dual_axis_repr_v5.train --smoke --output-dir outputs/dual_axis_repr_v5/smoke_test
    python -m experiments.representation_learning.dual_axis_repr_v5.train --episodes 20000 --output-dir outputs/dual_axis_repr_v5/run_default
"""

import argparse
import json
import os
from pathlib import Path
import time

from experiments.representation_learning.dual_axis_repr_v5.agent import DualAxisV5Agent
from experiments.representation_learning.dual_axis_repr_v5.trainer import DualAxisV5Trainer


def main():
    parser = argparse.ArgumentParser(
        description="Train dual-axis subspace-partitioned encoder for Leduc Hold'em")

    # Hyperparameters
    parser.add_argument('--lambda-hand', type=float, default=1.0,
                        help='Weight for hand SupCon loss (default 1.0)')
    parser.add_argument('--lambda-var', type=float, default=0.1,
                        help='Weight for VICReg variance loss (default 0.1)')
    parser.add_argument('--episodes', type=int, default=20000,
                        help='Total episodes to collect')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Adam learning rate')
    parser.add_argument('--batch-size', type=int, default=256,
                        help='Samples from replay buffer per update')

    # Output
    parser.add_argument('--output-dir', type=str,
                        default=str(Path(__file__).parent / 'outputs'),
                        help='Directory for checkpoints and results')

    # Misc
    parser.add_argument('--smoke', action='store_true',
                        help='Quick sanity check (200 episodes)')

    args = parser.parse_args()

    # Smoke test overrides
    if args.smoke:
        args.episodes = 200

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    save_path = os.path.join(args.output_dir, 'encoder.pt')

    print("=== dual_axis_repr_v5: Subspace-Partitioned Dual-Axis ===")
    print(f"Episodes: {args.episodes}, Batch: {args.batch_size}")
    print(f"lambda_hand: {args.lambda_hand}, lambda_var: {args.lambda_var}, lr: {args.lr}")
    print(f"Architecture: 15 -> 64 -> 64 -> 8")
    print(f"  dims 0:4 = reward subspace (L1 soft-distance loss only)")
    print(f"  dims 4:8 = hand subspace   (SupCon loss only)")
    print()

    # Build agent
    agent = DualAxisV5Agent()

    # Build trainer
    trainer = DualAxisV5Trainer(
        agent=agent,
        learning_rate=args.lr,
        batch_size=args.batch_size,
        lambda_hand=args.lambda_hand,
        lambda_var=args.lambda_var,
    )

    # Train
    t0 = time.time()
    loss_log = []

    def log_callback(info):
        if info.get('type') == 'batch_update':
            loss_log.append({
                'episode': info['episode'],
                'loss': info['loss'],
                'loss_reward': info.get('loss_reward', 0.0),
                'loss_hand': info.get('loss_hand', 0.0),
                'loss_var': info.get('loss_var', 0.0),
            })

    trainer.train(
        num_episodes=args.episodes,
        save_path=save_path,
        callback=log_callback,
    )

    elapsed = time.time() - t0
    print(f"\nTraining complete in {elapsed:.1f}s")
    print(f"Checkpoint saved to {save_path}")

    # Run post-training diagnostics
    print("\nRunning post-training diagnostics...")
    diagnostics = trainer.run_diagnostics()

    # Print v5 summary with subspace breakdown
    eff_dim         = diagnostics.get('effective_dim_80', 'N/A')
    rho_reward_sub  = diagnostics.get('reward_subspace_spearman_rho', 0)
    rho_full        = diagnostics.get('reward_spearman_rho_full', 0)
    hand_acc_full   = diagnostics.get('opp_hand_accuracy_full', 0)
    hand_acc_hand   = diagnostics.get('hand_subspace_opp_hand_accuracy', 0)

    print(f"\n=== dual_axis_repr_v5 (Subspace Partitioning) ===")
    print(f"Full embedding effective dim (80%): {eff_dim}")
    print(f"Reward subspace (dims 0:4) Spearman ρ: {rho_reward_sub:.3f}")
    print(f"Full embedding Spearman ρ: {rho_full:.3f}")
    print(f"Opp hand accuracy (full emb): {hand_acc_full:.3f}")
    print(f"Opp hand accuracy (hand subspace only): {hand_acc_hand:.3f}")
    print(f"================================================")

    # Additional cross-contamination diagnostics
    print(f"\n--- Cross-contamination check ---")
    print(f"Reward subspace hand accuracy (should be ~0.333 if isolated): "
          f"{diagnostics.get('reward_subspace_hand_accuracy', 0):.3f}")
    print(f"Hand subspace Spearman ρ (should be ~0 if isolated): "
          f"{diagnostics.get('hand_subspace_spearman_rho', 0):.3f}")
    print(f"Reward subspace eff dim (80%): {diagnostics.get('reward_subspace_effective_dim_80', 'N/A')}")
    print(f"Hand subspace eff dim (80%): {diagnostics.get('hand_subspace_effective_dim_80', 'N/A')}")

    # Save loss log
    log_path = os.path.join(args.output_dir, 'loss_log.json')
    with open(log_path, 'w') as f:
        json.dump(loss_log, f, indent=2)
    print(f"\nLoss log saved to {log_path}")

    # Save full results
    results = {
        'config': {
            'experiment': 'dual_axis_repr_v5',
            'approach': 'subspace_partitioning',
            'lambda_hand': args.lambda_hand,
            'lambda_var': args.lambda_var,
            'episodes': args.episodes,
            'lr': args.lr,
            'batch_size': args.batch_size,
        },
        'diagnostics': diagnostics,
        'training': {
            'elapsed_seconds': elapsed,
            'num_steps': len(loss_log),
            'final_loss': loss_log[-1]['loss'] if loss_log else None,
            'final_loss_reward': loss_log[-1].get('loss_reward') if loss_log else None,
            'final_loss_hand': loss_log[-1].get('loss_hand') if loss_log else None,
            'final_loss_var': loss_log[-1].get('loss_var') if loss_log else None,
        },
    }

    results_path = os.path.join(args.output_dir, 'results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {results_path}")


if __name__ == '__main__':
    main()
