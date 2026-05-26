"""Architecture-specific diagnostics for the relative-cap follow-up."""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.registry import registry
from engine.poker_session import PokerSession
from experiments.opp_encoder_modulation_v3_relative_cap.agent import OpponentEncoderModulationAgent


EXPERIMENT_ID = "opp_encoder_modulation_v3_relative_cap"
DEFAULT_OPPONENTS = ["heuristic", "value_based", "adaptive_value", "modulated_value", "cfr"]


def load_promoted_agent(agent_id: str):
    agent = registry.create(agent_id)
    checkpoint_path = registry.get_checkpoint_path(agent_id)
    if checkpoint_path and Path(checkpoint_path).exists():
        agent.load_model(checkpoint_path)
    agent.set_train_mode(False)
    return agent


def run_diagnostics(agent, opponent_id: str, num_hands: int) -> dict:
    opponent = load_promoted_agent(opponent_id)
    session = PokerSession()

    correct = 0
    total_predictions = 0
    gate_values = []
    abs_deltas = []
    abs_effective_deltas = []
    num_modulated_actions = 0

    for _ in range(num_hands):
        session.new_hand()
        while not session.is_finished:
            current_player = session.current_player
            if current_player == 0:
                obs = session.get_observation(viewer_id=0)
                evaluations = agent.get_action_evaluations(obs)
                selected = max(evaluations, key=lambda item: item["value"])
                if not (selected["is_terminal"] and selected["action"].name == "FOLD"):
                    gate_values.append(selected["gate"])
                    abs_deltas.append(abs(selected["delta"]))
                    abs_effective_deltas.append(abs(selected["gate"] * selected["delta"]))
                    num_modulated_actions += 1
                action = selected["action"]
            else:
                opponent_obs = session.get_observation(viewer_id=1)
                agent_view_obs = session.get_observation(viewer_id=0)
                state_encoding = agent.encode_observation(agent_view_obs, viewer_id=0)
                stats_vec = agent.encode_macro_stats(agent_view_obs)
                probs = agent.predict_action_probs(state_encoding, stats_vec).squeeze(0)
                predicted_action = int(probs.argmax().item())
                action = opponent.select_action(opponent_obs)
                correct += int(predicted_action == action.value)
                total_predictions += 1
            session.step(action)

    accuracy = correct / total_predictions if total_predictions else 0.0
    mean_gate = sum(gate_values) / len(gate_values) if gate_values else 0.0
    mean_abs_delta = sum(abs_deltas) / len(abs_deltas) if abs_deltas else 0.0
    mean_abs_effective_delta = (
        sum(abs_effective_deltas) / len(abs_effective_deltas) if abs_effective_deltas else 0.0
    )

    return {
        "action_prediction_accuracy": round(accuracy, 4),
        "mean_gate": round(mean_gate, 4),
        "mean_abs_delta": round(mean_abs_delta, 4),
        "mean_abs_effective_delta": round(mean_abs_effective_delta, 4),
        "num_predictions": total_predictions,
        "num_modulated_actions": num_modulated_actions,
    }


def main():
    parser = argparse.ArgumentParser(description="Diagnose opponent-encoder modulation v3 relative cap.")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=ROOT / "outputs" / EXPERIMENT_ID / "checkpoint.pt",
        help="Experiment checkpoint path.",
    )
    parser.add_argument("--hands", type=int, default=1000, help="Hands per diagnostic opponent.")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs" / EXPERIMENT_ID / "diagnostics.json",
        help="Diagnostic JSON path.",
    )
    args = parser.parse_args()

    agent = OpponentEncoderModulationAgent(model_path=str(args.checkpoint))
    agent.set_train_mode(False)

    payload = {
        "experiment_id": EXPERIMENT_ID,
        "checkpoint": str(args.checkpoint),
        "per_opponent": {opponent_id: run_diagnostics(agent, opponent_id, args.hands) for opponent_id in DEFAULT_OPPONENTS},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
