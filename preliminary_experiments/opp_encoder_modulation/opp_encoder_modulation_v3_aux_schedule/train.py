"""Train the opponent-encoder modulation auxiliary-schedule experiment."""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.opp_encoder_modulation_v3_aux_schedule.agent import OpponentEncoderModulationAgent
from experiments.opp_encoder_modulation_v3_aux_schedule.trainer import OpponentEncoderModulationTrainer


EXPERIMENT_ID = "opp_encoder_modulation_v3_aux_schedule"


def build_output_paths(output_dir: Path) -> dict:
    return {
        "checkpoint": output_dir / "checkpoint.pt",
        "history": output_dir / "train_history.json",
        "config": output_dir / "train_config.json",
    }


def main():
    parser = argparse.ArgumentParser(description="Train opponent-encoder modulation v3 auxiliary schedule.")
    parser.add_argument("--sessions", type=int, default=40000, help="Training sessions.")
    parser.add_argument("--batch-size", type=int, default=128, help="Hands per update.")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate.")
    parser.add_argument("--hands-per-session", type=int, default=30, help="Hands per session.")
    parser.add_argument("--action-loss-weight", type=float, default=0.5, help="Target auxiliary loss weight.")
    parser.add_argument("--aux-warmup-sessions", type=int, default=10000, help="Sessions before action loss turns on.")
    parser.add_argument("--aux-ramp-sessions", type=int, default=10000, help="Sessions used to ramp action loss up.")
    parser.add_argument("--gate-target", type=float, default=0.4, help="Target average gate value.")
    parser.add_argument("--gate-reg-weight", type=float, default=0.5, help="Weight on gate-target regularization.")
    parser.add_argument(
        "--modulation-reg-weight",
        type=float,
        default=0.5,
        help="Weight on effective modulation regularization.",
    )
    parser.add_argument("--rotate-every", type=int, default=200, help="Rotate opponents every N sessions.")
    parser.add_argument("--snapshot-every", type=int, default=2000, help="Add self snapshot every N sessions.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "outputs" / EXPERIMENT_ID,
        help="Ignored output directory.",
    )
    parser.add_argument(
        "--base-model-path",
        type=Path,
        default=Path(OpponentEncoderModulationAgent.default_base_model_path()),
        help="Frozen base checkpoint path.",
    )
    parser.add_argument("--smoke", action="store_true", help="Use a tiny budget for a quick pipeline check.")
    args = parser.parse_args()

    if args.smoke:
        args.sessions = 20
        args.batch_size = 16
        args.hands_per_session = 5
        args.rotate_every = 5
        args.snapshot_every = 1000
        args.aux_warmup_sessions = 5
        args.aux_ramp_sessions = 5

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_paths = build_output_paths(args.output_dir)

    config = {
        "experiment_id": EXPERIMENT_ID,
        "sessions": args.sessions,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "hands_per_session": args.hands_per_session,
        "target_action_loss_weight": args.action_loss_weight,
        "aux_warmup_sessions": args.aux_warmup_sessions,
        "aux_ramp_sessions": args.aux_ramp_sessions,
        "gate_target": args.gate_target,
        "gate_reg_weight": args.gate_reg_weight,
        "modulation_reg_weight": args.modulation_reg_weight,
        "rotate_every": args.rotate_every,
        "snapshot_every": args.snapshot_every,
        "base_model_path": str(args.base_model_path),
    }
    output_paths["config"].write_text(json.dumps(config, indent=2))

    agent = OpponentEncoderModulationAgent(base_model_path=str(args.base_model_path))
    trainer = OpponentEncoderModulationTrainer(
        agent,
        learning_rate=args.lr,
        hands_per_session=args.hands_per_session,
        action_loss_weight=args.action_loss_weight,
        aux_warmup_sessions=args.aux_warmup_sessions,
        aux_ramp_sessions=args.aux_ramp_sessions,
        gate_target=args.gate_target,
        gate_reg_weight=args.gate_reg_weight,
        modulation_reg_weight=args.modulation_reg_weight,
        rotate_every=args.rotate_every,
        snapshot_every=args.snapshot_every,
    )

    history = []

    def callback(event):
        history.append(event)
        output_paths["history"].write_text(json.dumps(history, indent=2))

    trainer.train(
        num_episodes=args.sessions,
        batch_size=args.batch_size,
        save_path=str(output_paths["checkpoint"]),
        callback=callback,
    )


if __name__ == "__main__":
    main()
