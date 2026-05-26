"""Training entrypoint for value_dim_search_v1.

Usage:
    # Run A: 32x32 architecture
    python -m experiments.representation_learning.value_dim_search_v1.train --hidden-dims 32 32 --output-dir outputs/value_dim_search_v1/run_A_32x32

    # Run B: 32x16 architecture
    python -m experiments.representation_learning.value_dim_search_v1.train --hidden-dims 32 16 --output-dir outputs/value_dim_search_v1/run_B_32x16

    # Smoke test
    python -m experiments.representation_learning.value_dim_search_v1.train --hidden-dims 32 32 --smoke --output-dir outputs/value_dim_search_v1/smoke_test
"""

import argparse
import json
import os
from pathlib import Path
import time

from agents.evaluation import evaluate_agents
from agents.heuristic.agent import HeuristicAgent

from experiments.representation_learning.value_dim_search_v1.agent import ValueDimAgent
from experiments.representation_learning.value_dim_search_v1.trainer import ValueDimTrainer


def main():
    parser = argparse.ArgumentParser(
        description="Train ValueDimAgent with configurable hidden dimensions.")

    parser.add_argument('--hidden-dims', nargs='+', type=int, default=[32, 32],
                        help='Hidden layer sizes, e.g. --hidden-dims 32 32')
    parser.add_argument('--episodes', type=int, default=20000,
                        help='Number of training episodes')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Adam learning rate')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Episodes per network update')
    parser.add_argument('--output-dir', type=str,
                        default=str(Path(__file__).parent / 'outputs' / 'default'),
                        help='Directory for checkpoint and logs')
    parser.add_argument('--smoke', action='store_true',
                        help='Quick sanity check (200 episodes)')

    args = parser.parse_args()

    # Smoke test overrides
    if args.smoke:
        args.episodes = 200

    os.makedirs(args.output_dir, exist_ok=True)

    arch_str = 'x'.join(str(h) for h in args.hidden_dims)
    print(f"=== value_dim_search_v1 | arch: 15->{arch_str}->1 ===")
    print(f"Episodes: {args.episodes}, LR: {args.lr}, Batch: {args.batch_size}")
    print(f"Output dir: {args.output_dir}")
    print()

    # Build agent and trainer
    agent = ValueDimAgent(hidden_dims=args.hidden_dims)
    trainer = ValueDimTrainer(agent, learning_rate=args.lr)

    # Count parameters
    num_params = sum(p.numel() for p in agent.model.parameters())
    print(f"Network parameters: {num_params:,}")
    print()

    # Training history
    train_history = []

    def log_callback(info):
        train_history.append(info)

    checkpoint_path = os.path.join(args.output_dir, 'checkpoint.pt')

    t0 = time.time()
    trainer.train(
        num_episodes=args.episodes,
        batch_size=args.batch_size,
        save_path=checkpoint_path,
        callback=log_callback,
    )
    elapsed = time.time() - t0
    print(f"\nTraining complete in {elapsed:.1f}s")

    # Save training history
    history_path = os.path.join(args.output_dir, 'train_history.json')
    with open(history_path, 'w') as f:
        json.dump(train_history, f, indent=2)
    print(f"Training history saved to {history_path}")

    # Final evaluation vs heuristic
    print("\nRunning final evaluation vs HeuristicAgent (500 rounds)...")
    agent.set_train_mode(False)
    heuristic = HeuristicAgent()
    result = evaluate_agents(agent, heuristic, num_rounds=500)
    avg_chips = result.agent_0_avg_chips

    print(f"Final avg chips/round vs heuristic: {avg_chips:+.4f}")
    print(f"(agent_0={result.agent_0_avg_chips:+.4f}, agent_1={result.agent_1_avg_chips:+.4f})")

    # Save run metadata
    run_meta = {
        "hidden_dims": args.hidden_dims,
        "arch_str": f"15->{arch_str}->1",
        "num_params": num_params,
        "episodes": args.episodes,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "elapsed_seconds": round(elapsed, 1),
        "avg_chips_vs_heuristic": round(avg_chips, 4),
        "eval_rounds": result.num_rounds,
        "smoke": args.smoke,
    }

    meta_path = os.path.join(args.output_dir, 'run_meta.json')
    with open(meta_path, 'w') as f:
        json.dump(run_meta, f, indent=2)
    print(f"Run metadata saved to {meta_path}")


if __name__ == '__main__':
    main()
