"""Training entrypoint for repr_policy_v1 experiment.

Usage:
    # Baseline (no encoder checkpoint needed)
    python -m experiments.representation_learning.repr_policy_v1.train --variant baseline --episodes 20000 \
        --output-dir outputs/repr_policy_v1/run_baseline

    # Frozen encoder
    python -m experiments.representation_learning.repr_policy_v1.train --variant frozen \
        --checkpoint-path outputs/contrastive_repr_v1/encoder_l1.pt \
        --episodes 20000 --output-dir outputs/repr_policy_v1/run_frozen

    # Fine-tuned encoder
    python -m experiments.representation_learning.repr_policy_v1.train --variant finetune \
        --checkpoint-path outputs/contrastive_repr_v1/encoder_l1.pt \
        --episodes 20000 --output-dir outputs/repr_policy_v1/run_finetune

    # Smoke test (200 episodes)
    python -m experiments.representation_learning.repr_policy_v1.train --variant baseline --smoke \
        --output-dir outputs/repr_policy_v1/smoke_baseline
"""

import argparse
from pathlib import Path
import json
import os

from experiments.representation_learning.repr_policy_v1.agent import (
    VanillaPolicyAgent,
    ReprPolicyAgent,
    ReprPolicyFineTuneAgent,
)
from experiments.representation_learning.repr_policy_v1.trainer import REINFORCETrainer
from agents.evaluation import evaluate_agents
from agents.heuristic.agent import HeuristicAgent


def parse_args():
    parser = argparse.ArgumentParser(description="Train repr_policy_v1 agent")
    parser.add_argument(
        "--variant",
        choices=["baseline", "frozen", "finetune"],
        default="baseline",
        help="Which agent variant to train",
    )
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        default=None,
        help="Path to contrastive encoder checkpoint (required for frozen/finetune)",
    )
    parser.add_argument("--episodes", type=int, default=20000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--output-dir", type=str, default=str(Path(__file__).parent / 'outputs'))
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Smoke test mode: only 200 episodes",
    )
    return parser.parse_args()


def build_agent(variant: str, checkpoint_path: str):
    if variant == "baseline":
        return VanillaPolicyAgent()
    elif variant == "frozen":
        if not checkpoint_path:
            raise ValueError("--checkpoint-path is required for variant=frozen")
        return ReprPolicyAgent(checkpoint_path=checkpoint_path)
    elif variant == "finetune":
        if not checkpoint_path:
            raise ValueError("--checkpoint-path is required for variant=finetune")
        return ReprPolicyFineTuneAgent(checkpoint_path=checkpoint_path)
    else:
        raise ValueError(f"Unknown variant: {variant}")


def main():
    args = parse_args()

    if args.smoke:
        args.episodes = 200
        print("[SMOKE TEST] Running 200 episodes only.")

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Building agent: variant={args.variant}")
    agent = build_agent(args.variant, args.checkpoint_path)

    trainer = REINFORCETrainer(agent, learning_rate=args.lr)

    # Training history for logging
    train_history = []

    def callback(info):
        train_history.append(info)

    checkpoint_path = os.path.join(args.output_dir, "checkpoint.pt")
    history_path = os.path.join(args.output_dir, "train_history.json")

    print(f"Training {args.variant} for {args.episodes} episodes "
          f"(batch_size={args.batch_size}, lr={args.lr})")

    trainer.train(
        num_episodes=args.episodes,
        batch_size=args.batch_size,
        save_path=checkpoint_path,
        callback=callback,
    )

    # Save training history
    with open(history_path, "w") as f:
        json.dump(train_history, f, indent=2)
    print(f"Training history saved to {history_path}")

    # Final evaluation vs heuristic
    print("\nEvaluating vs HeuristicAgent (1000 rounds)...")
    agent.set_train_mode(False)
    heuristic = HeuristicAgent()
    result = evaluate_agents(agent, heuristic, num_rounds=1000)

    print(f"\n=== Final Evaluation ===")
    print(f"Variant: {args.variant}")
    print(f"Avg chips vs heuristic: {result.agent_0_avg_chips:+.4f}")
    print(f"Total rounds: {result.num_rounds}")

    # Save final eval result
    eval_result = {
        "variant": args.variant,
        "episodes_trained": args.episodes,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "checkpoint_path": args.checkpoint_path,
        "avg_chips_vs_heuristic": result.agent_0_avg_chips,
        "num_eval_rounds": result.num_rounds,
    }
    eval_path = os.path.join(args.output_dir, "eval_result.json")
    with open(eval_path, "w") as f:
        json.dump(eval_result, f, indent=2)
    print(f"Eval result saved to {eval_path}")


if __name__ == "__main__":
    main()
