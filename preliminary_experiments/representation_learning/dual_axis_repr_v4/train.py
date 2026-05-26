"""Training entrypoint for dual-axis EMA-normalized representation learning (v4).

Combines L1 soft-distance reward loss (from contrastive_repr_v1) with
SupCon hand-identity loss (from dual_axis_repr_v2), with EMA normalization
to equalize gradient contributions (fixing the 2,600x scale imbalance in v3).

Usage:
    python -m experiments.representation_learning.dual_axis_repr_v4.train --smoke --output-dir outputs/dual_axis_repr_v4/smoke_test
    python -m experiments.representation_learning.dual_axis_repr_v4.train --episodes 20000 --output-dir outputs/dual_axis_repr_v4/run_default
"""

import argparse
import json
import os
from pathlib import Path
import time

from experiments.representation_learning.dual_axis_repr_v4.agent import DualAxisV4Agent
from experiments.representation_learning.dual_axis_repr_v4.trainer import DualAxisV4Trainer


def main():
    parser = argparse.ArgumentParser(
        description="Train dual-axis EMA-normalized encoder for Leduc Hold'em")

    # Hyperparameters
    parser.add_argument('--lambda-hand', type=float, default=1.0,
                        help='Weight for hand SupCon loss (default 1.0)')
    parser.add_argument('--lambda-var', type=float, default=0.1,
                        help='Weight for VICReg variance loss (default 0.1)')
    parser.add_argument('--ema-alpha', type=float, default=0.99,
                        help='EMA smoothing factor for loss normalization (default 0.99)')
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

    print("=== dual_axis_repr_v4: EMA-Normalized Hybrid L1 + SupCon(hand) ===")
    print(f"Episodes: {args.episodes}, Batch: {args.batch_size}")
    print(f"lambda_hand: {args.lambda_hand}, lambda_var: {args.lambda_var}, "
          f"ema_alpha: {args.ema_alpha}, lr: {args.lr}")
    print()

    # Build agent
    agent = DualAxisV4Agent()

    # Build trainer
    trainer = DualAxisV4Trainer(
        agent=agent,
        learning_rate=args.lr,
        batch_size=args.batch_size,
        lambda_hand=args.lambda_hand,
        lambda_var=args.lambda_var,
        ema_alpha=args.ema_alpha,
    )

    # Train
    t0 = time.time()
    loss_log = []

    def log_callback(info):
        if info.get('type') == 'batch_update':
            loss_log.append({
                'episode': info['episode'],
                'loss': info['loss'],
                'loss_reward_raw': info.get('loss_reward_raw', 0.0),
                'loss_hand_raw': info.get('loss_hand_raw', 0.0),
                'loss_var': info.get('loss_var', 0.0),
                'ema_reward': info.get('ema_reward', 0.0),
                'ema_hand': info.get('ema_hand', 0.0),
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

    # Retrieve final EMA values
    final_ema_reward = diagnostics.get('final_ema_reward', 0.0)
    final_ema_hand = diagnostics.get('final_ema_hand', 0.0)

    # Print summary
    print(f"\n=== dual_axis_repr_v4 Results ===")
    print(f"Effective dimension (80%): {diagnostics.get('effective_dim_80', 'N/A')}")
    print(f"Opp hand linear probe: {diagnostics.get('opp_hand_accuracy', 0):.3f} (chance=0.333)")
    print(f"Reward Spearman rho: {diagnostics.get('reward_spearman_rho', 0):.3f}")
    print(f"Reward bin accuracy: {diagnostics.get('reward_bin_accuracy', 0):.3f} (chance=0.200)")
    print(f"EMA at end — L_reward: {final_ema_reward:.3f}, L_hand: {final_ema_hand:.3f}")
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
            'ema_alpha': args.ema_alpha,
            'episodes': args.episodes,
            'lr': args.lr,
            'batch_size': args.batch_size,
        },
        'diagnostics': diagnostics,
        'training': {
            'elapsed_seconds': elapsed,
            'num_steps': len(loss_log),
            'final_loss': loss_log[-1]['loss'] if loss_log else None,
            'final_loss_reward_raw': loss_log[-1].get('loss_reward_raw') if loss_log else None,
            'final_loss_hand_raw': loss_log[-1].get('loss_hand_raw') if loss_log else None,
            'final_loss_var': loss_log[-1].get('loss_var') if loss_log else None,
            'final_ema_reward': loss_log[-1].get('ema_reward') if loss_log else None,
            'final_ema_hand': loss_log[-1].get('ema_hand') if loss_log else None,
        },
    }

    results_path = os.path.join(args.output_dir, 'results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {results_path}")


if __name__ == '__main__':
    main()
