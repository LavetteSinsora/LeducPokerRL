"""Evaluate the state-gated follow-up against promoted agents."""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.evaluation import compute_robustness_metrics, evaluate_agents
from agents.registry import registry
from experiments.opp_encoder_modulation_v1.agent import (
    OpponentEncoderModulationAgent as OpponentEncoderModulationAgentV1,
)
from experiments.opp_encoder_modulation_v2.agent import OpponentEncoderModulationAgent as OpponentEncoderModulationAgentV2
from experiments.opp_encoder_modulation_v3_state_gate.agent import OpponentEncoderModulationAgent


EXPERIMENT_ID = "opp_encoder_modulation_v3_state_gate"
DEFAULT_OPPONENTS = ["heuristic", "value_based", "adaptive_value", "modulated_value", "cfr"]


def load_promoted_agent(agent_id: str):
    agent = registry.create(agent_id)
    checkpoint_path = registry.get_checkpoint_path(agent_id)
    if checkpoint_path and Path(checkpoint_path).exists():
        agent.load_model(checkpoint_path)
    agent.set_train_mode(False)
    return agent


def evaluate_suite(agent, opponents, rounds_per_matchup: int):
    scores = {}
    details = {}
    for opponent_id in opponents:
        opponent = load_promoted_agent(opponent_id)
        result = evaluate_agents(agent, opponent, num_rounds=rounds_per_matchup)
        scores[opponent_id] = result.agent_0_avg_chips
        details[opponent_id] = {
            "avg_chips_per_round": result.agent_0_avg_chips,
            "total_chips": result.agent_0_total_chips,
            "rounds": result.num_rounds,
        }
    return scores, details


def main():
    parser = argparse.ArgumentParser(description="Evaluate opponent-encoder modulation v3 state gate.")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=ROOT / "outputs" / EXPERIMENT_ID / "checkpoint.pt",
        help="Experiment checkpoint path.",
    )
    parser.add_argument("--rounds", type=int, default=1000, help="Rounds per matchup.")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs" / EXPERIMENT_ID / "evaluation.json",
        help="Evaluation JSON path.",
    )
    args = parser.parse_args()

    agent = OpponentEncoderModulationAgent(model_path=str(args.checkpoint))
    agent.set_train_mode(False)
    scores, details = evaluate_suite(agent, DEFAULT_OPPONENTS, args.rounds)

    control = load_promoted_agent("modulated_value")
    control_scores, control_details = evaluate_suite(control, DEFAULT_OPPONENTS, args.rounds)

    v1_agent = OpponentEncoderModulationAgentV1(
        model_path=str(ROOT / "outputs" / "opp_encoder_modulation_v1" / "checkpoint.pt")
    )
    v1_agent.set_train_mode(False)
    v1_scores, v1_details = evaluate_suite(v1_agent, DEFAULT_OPPONENTS, args.rounds)

    v2_agent = OpponentEncoderModulationAgentV2(
        model_path=str(ROOT / "outputs" / "opp_encoder_modulation_v2" / "checkpoint.pt")
    )
    v2_agent.set_train_mode(False)
    v2_scores, v2_details = evaluate_suite(v2_agent, DEFAULT_OPPONENTS, args.rounds)

    payload = {
        "experiment_id": EXPERIMENT_ID,
        "checkpoint": str(args.checkpoint),
        "candidate": {"scores": scores, "metrics": compute_robustness_metrics(scores), "details": details},
        "control_modulated_value": {
            "scores": control_scores,
            "metrics": compute_robustness_metrics(control_scores),
            "details": control_details,
        },
        "baseline_v1": {"scores": v1_scores, "metrics": compute_robustness_metrics(v1_scores), "details": v1_details},
        "baseline_v2": {"scores": v2_scores, "metrics": compute_robustness_metrics(v2_scores), "details": v2_details},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
