"""Train the hand-conditioned opponent action model experiment."""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.hand_conditioned_action_model_v1.agent import HandConditionedActionModel
from experiments.hand_conditioned_action_model_v1.trainer import (
    DEFAULT_EVAL_POOL,
    DEFAULT_TRAIN_POOL,
    HandConditionedActionModelTrainer,
)


EXPERIMENT_ID = "hand_conditioned_action_model_v1"


def build_output_paths(output_dir: Path) -> dict:
    return {
        "checkpoint": output_dir / "checkpoint.pt",
        "history": output_dir / "train_history.json",
        "config": output_dir / "train_config.json",
    }


def parse_pool(arg: str, default):
    if not arg:
        return tuple(default)
    return tuple(item.strip() for item in arg.split(",") if item.strip())


def main():
    parser = argparse.ArgumentParser(description="Train the hand-conditioned action model.")
    parser.add_argument("--sessions", type=int, default=40000, help="Training sessions.")
    parser.add_argument("--batch-size", type=int, default=64, help="Sessions per update.")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate.")
    parser.add_argument("--hands-per-session", type=int, default=30, help="Hands per session.")
    parser.add_argument("--hidden-size", type=int, default=64, help="Hidden layer width.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument("--train-pool", type=str, default=",".join(DEFAULT_TRAIN_POOL), help="Comma-separated training agents.")
    parser.add_argument("--eval-pool", type=str, default=",".join(DEFAULT_EVAL_POOL), help="Comma-separated evaluation agents.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "outputs" / EXPERIMENT_ID,
        help="Ignored output directory.",
    )
    parser.add_argument("--smoke", action="store_true", help="Use a tiny budget for a quick pipeline check.")
    args = parser.parse_args()

    if args.smoke:
        args.sessions = 20
        args.batch_size = 8
        args.hands_per_session = 5

    train_pool = parse_pool(args.train_pool, DEFAULT_TRAIN_POOL)
    eval_pool = parse_pool(args.eval_pool, DEFAULT_EVAL_POOL)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_paths = build_output_paths(args.output_dir)

    config = {
        "experiment_id": EXPERIMENT_ID,
        "sessions": args.sessions,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "hands_per_session": args.hands_per_session,
        "hidden_size": args.hidden_size,
        "seed": args.seed,
        "train_pool": list(train_pool),
        "eval_pool": list(eval_pool),
    }
    output_paths["config"].write_text(json.dumps(config, indent=2))

    model = HandConditionedActionModel(hidden_size=args.hidden_size)
    trainer = HandConditionedActionModelTrainer(
        model,
        learning_rate=args.lr,
        hands_per_session=args.hands_per_session,
        train_pool_ids=train_pool,
        eval_pool_ids=eval_pool,
        seed=args.seed,
    )

    history = []

    def callback(event):
        history.append(event)
        output_paths["history"].write_text(json.dumps(history, indent=2))

    trainer.train(
        num_sessions=args.sessions,
        batch_size=args.batch_size,
        save_path=str(output_paths["checkpoint"]),
        callback=callback,
    )


if __name__ == "__main__":
    main()
