"""Training entrypoint for contrastive representation learning.

Usage:
    python -m experiments.representation_learning.contrastive_repr_v1.train --loss-type L1
    python -m experiments.representation_learning.contrastive_repr_v1.train --loss-type L2 --temperature 0.3
    python -m experiments.representation_learning.contrastive_repr_v1.train --smoke          # quick sanity check
"""

import argparse
import json
import os
from pathlib import Path
import time

from experiments.representation_learning.contrastive_repr_v1.agent import ContrastiveReprAgent
from experiments.representation_learning.contrastive_repr_v1.trainer import ContrastiveTrainer


def main():
    parser = argparse.ArgumentParser(
        description="Train contrastive state encoder for Leduc Hold'em")

    # Core settings
    parser.add_argument('--loss-type', type=str, default='L1',
                        choices=['L0', 'L1', 'L2'],
                        help='Loss formulation: L0 (TD control), L1 (distance), L2 (RnC)')
    parser.add_argument('--episodes', type=int, default=20000,
                        help='Total episodes to collect')
    parser.add_argument('--episodes-per-step', type=int, default=8,
                        help='Episodes collected per training step')
    parser.add_argument('--contrastive-batch-size', type=int, default=256,
                        help='Samples from replay buffer per contrastive update')
    parser.add_argument('--buffer-capacity', type=int, default=5000,
                        help='Replay buffer capacity')

    # Hyperparameters
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--beta', type=float, default=None,
                        help='L1 distance exchange rate (None = auto-calibrate)')
    parser.add_argument('--temperature', type=float, default=0.5,
                        help='L2 RnC temperature')
    parser.add_argument('--lambda-var', type=float, default=0.1,
                        help='VICReg variance weight (L1 only)')
    parser.add_argument('--embedding-dim', type=int, default=8,
                        help='Embedding dimensionality')

    # Data collection
    parser.add_argument('--data-agent-path', type=str,
                        default='agents/value_based/checkpoint.pt',
                        help='Path to trained value-based agent checkpoint')
    parser.add_argument('--cross-trajectory-only', action='store_true',
                        help='Exclude same-episode pairs from contrastive loss')

    # Output
    parser.add_argument('--output-dir', type=str,
                        default=str(Path(__file__).parent / 'outputs'),
                        help='Directory for checkpoints and logs')
    parser.add_argument('--smoke', action='store_true',
                        help='Quick sanity check (100 episodes, small batch)')

    args = parser.parse_args()

    # Smoke test overrides
    if args.smoke:
        args.episodes = 100
        args.contrastive_batch_size = 64
        args.buffer_capacity = 500
        args.episodes_per_step = 4

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Determine suffix for this run
    suffix = args.loss_type.lower()
    if args.cross_trajectory_only:
        suffix += '_cross_traj'
    save_path = os.path.join(args.output_dir, f'encoder_{suffix}.pt')

    print(f"=== Contrastive Repr v1: {args.loss_type} ===")
    print(f"Episodes: {args.episodes}, Batch: {args.contrastive_batch_size}, "
          f"Buffer: {args.buffer_capacity}")
    if args.loss_type == 'L1':
        beta_str = f"{args.beta:.4f}" if args.beta else "auto"
        print(f"Beta: {beta_str}, Lambda_var: {args.lambda_var}")
    elif args.loss_type == 'L2':
        print(f"Temperature: {args.temperature}")
    if args.cross_trajectory_only:
        print("Cross-trajectory only: ON")
    print()

    # Build agent
    use_value_head = (args.loss_type == 'L0')
    agent = ContrastiveReprAgent(
        embedding_dim=args.embedding_dim,
        use_value_head=use_value_head,
    )

    # Build trainer
    trainer = ContrastiveTrainer(
        agent=agent,
        data_agent_path=args.data_agent_path,
        loss_type=args.loss_type,
        learning_rate=args.lr,
        contrastive_batch_size=args.contrastive_batch_size,
        buffer_capacity=args.buffer_capacity,
        lambda_var=args.lambda_var,
        temperature=args.temperature,
        beta=args.beta,
        cross_trajectory_only=args.cross_trajectory_only,
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

    batch_size = args.episodes_per_step if args.loss_type != 'L0' else 32
    trainer.train(
        num_episodes=args.episodes,
        batch_size=batch_size,
        save_path=save_path,
        callback=log_callback,
        episodes_per_step=args.episodes_per_step,
    )

    elapsed = time.time() - t0
    print(f"\nTraining complete in {elapsed:.1f}s")
    print(f"Checkpoint saved to {save_path}")

    # Save loss log
    log_path = os.path.join(args.output_dir, f'loss_log_{suffix}.json')
    with open(log_path, 'w') as f:
        json.dump(loss_log, f)
    print(f"Loss log saved to {log_path}")


if __name__ == '__main__':
    main()
