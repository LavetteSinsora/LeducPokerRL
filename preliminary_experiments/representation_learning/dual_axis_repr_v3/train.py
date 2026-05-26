"""Training entrypoint for dual-axis hybrid representation learning (v3).

Combines L1 soft-distance reward loss (from contrastive_repr_v1) with
SupCon hand-identity loss (from dual_axis_repr_v2).

Usage:
    python -m experiments.representation_learning.dual_axis_repr_v3.train --smoke --output-dir outputs/dual_axis_repr_v3/smoke_test
    python -m experiments.representation_learning.dual_axis_repr_v3.train --episodes 20000 --output-dir outputs/dual_axis_repr_v3/run_default
"""

import argparse
import json
import os
from pathlib import Path
import time

from experiments.representation_learning.dual_axis_repr_v3.agent import DualAxisV3Agent
from experiments.representation_learning.dual_axis_repr_v3.trainer import DualAxisV3Trainer


def main():
    parser = argparse.ArgumentParser(
        description="Train dual-axis hybrid encoder for Leduc Hold'em")

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

    print("=== dual_axis_repr_v3: Hybrid L1 + SupCon(hand) ===")
    print(f"Episodes: {args.episodes}, Batch: {args.batch_size}")
    print(f"lambda_hand: {args.lambda_hand}, lambda_var: {args.lambda_var}, lr: {args.lr}")
    print()

    # Build agent
    agent = DualAxisV3Agent()

    # Build trainer
    trainer = DualAxisV3Trainer(
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

    # Print summary
    print(f"\n=== dual_axis_repr_v3 Results ===")
    print(f"Effective dimension (80%): {diagnostics.get('effective_dim_80', 'N/A')}")
    print(f"Opp hand linear probe: {diagnostics.get('opp_hand_accuracy', 0):.3f} (chance=0.333)")
    print(f"Reward Spearman rho: {diagnostics.get('reward_spearman_rho', 0):.3f}")
    print(f"Reward bin accuracy: {diagnostics.get('reward_bin_accuracy', 0):.3f} (chance=0.200)")
    print(f"=================================")

    # Save loss log
    log_path = os.path.join(args.output_dir, 'loss_log.json')
    with open(log_path, 'w') as f:
        json.dump(loss_log, f, indent=2)
    print(f"\nLoss log saved to {log_path}")

    # Save diagnostics/results
    results = {
        'config': {
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
