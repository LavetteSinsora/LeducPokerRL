"""Training entrypoint for dual-axis SupCon representation learning.

Usage:
    python -m experiments.representation_learning.dual_axis_repr_v2.train
    python -m experiments.representation_learning.dual_axis_repr_v2.train --smoke --output-dir outputs/dual_axis_repr_v2/smoke_test
    python -m experiments.representation_learning.dual_axis_repr_v2.train --episodes 20000 --output-dir outputs/dual_axis_repr_v2/run_default
"""

import argparse
import json
import os
from pathlib import Path
import time

from experiments.representation_learning.dual_axis_repr_v2.agent import DualAxisV2Agent
from experiments.representation_learning.dual_axis_repr_v2.trainer import DualAxisV2Trainer


def main():
    parser = argparse.ArgumentParser(
        description="Train dual-axis SupCon encoder for Leduc Hold'em")

    # SupCon hyperparameters
    parser.add_argument('--temperature', type=float, default=0.07,
                        help='SupCon temperature tau (default 0.07, standard SupCon)')
    parser.add_argument('--lambda-hand', type=float, default=1.0,
                        help='Weight for hand SupCon loss')
    parser.add_argument('--lambda-var', type=float, default=0.1,
                        help='Weight for VICReg variance loss')
    parser.add_argument('--n-bins', type=int, default=5,
                        help='Number of reward discretization bins')

    # Training settings
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

    print("=== dual_axis_repr_v2: Dual-Axis SupCon ===")
    print(f"Episodes: {args.episodes}, Batch: {args.batch_size}")
    print(f"Temperature: {args.temperature}, lambda_hand: {args.lambda_hand}, "
          f"lambda_var: {args.lambda_var}")
    print(f"Reward bins: {args.n_bins}")
    print()

    # Build agent
    agent = DualAxisV2Agent()

    # Build trainer
    trainer = DualAxisV2Trainer(
        agent=agent,
        learning_rate=args.lr,
        batch_size=args.batch_size,
        temperature=args.temperature,
        lambda_hand=args.lambda_hand,
        lambda_var=args.lambda_var,
        n_bins=args.n_bins,
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
    print(f"\n=== dual_axis_repr_v2 Results ===")
    print(f"Effective dimension (80%): {diagnostics.get('effective_dim_80', 'N/A')}")
    print(f"Opp hand linear probe: {diagnostics.get('opp_hand_accuracy', 0):.3f} (chance=0.333)")
    print(f"Reward Spearman rho (pairwise): {diagnostics.get('reward_spearman_rho', 0):.3f}")
    print(f"================================")

    # Save loss log
    log_path = os.path.join(args.output_dir, 'loss_log.json')
    with open(log_path, 'w') as f:
        json.dump(loss_log, f, indent=2)
    print(f"\nLoss log saved to {log_path}")

    # Save diagnostics/results
    results = {
        'config': {
            'temperature': args.temperature,
            'lambda_hand': args.lambda_hand,
            'lambda_var': args.lambda_var,
            'n_bins': args.n_bins,
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
