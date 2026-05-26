"""Evaluate the hand-conditioned action model on held-out opponents."""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine.poker_session import PokerSession
from experiments.hand_conditioned_action_model_v1.agent import (
    HandConditionedActionModel,
    initialize_belief,
    update_belief_with_board,
)
from experiments.hand_conditioned_action_model_v1.trainer import DEFAULT_EVAL_POOL, load_registry_agent


EXPERIMENT_ID = "hand_conditioned_action_model_v1"


def evaluate_target(model, target_id: str, probe_id: str, num_sessions: int, hands_per_session: int) -> dict:
    target_agent = load_registry_agent(target_id)
    probe_agent = load_registry_agent(probe_id)
    pairings = [(target_agent, probe_agent), (probe_agent, target_agent)]

    total_actions = 0
    correct_actions = 0
    total_true_hand_prob = 0.0
    total_prior_true_hand_prob = 0.0
    total_belief_top1 = 0.0
    total_prior_top1 = 0.0
    total_tvd_shift = 0.0

    for left, right in pairings:
        for _ in range(num_sessions):
            session = PokerSession()
            for _ in range(hands_per_session):
                session.new_hand()
                beliefs = [
                    initialize_belief(session.game.player_hands[0]),
                    initialize_belief(session.game.player_hands[1]),
                ]
                board_seen = [None, None]

                while not session.is_finished:
                    actor = session.current_player
                    viewer = 1 - actor
                    agents = [left, right]
                    actor_obs = session.get_observation(viewer_id=actor)
                    observer_obs = session.get_observation(viewer_id=viewer)
                    action = agents[actor].select_action(actor_obs)
                    if isinstance(action, tuple):
                        action = action[0]

                    if agents[actor] is target_agent:
                        if observer_obs.board is not None and board_seen[viewer] is None:
                            beliefs[viewer] = update_belief_with_board(
                                beliefs[viewer],
                                viewer_hand=observer_obs.player_hand,
                                board=observer_obs.board,
                            )
                            board_seen[viewer] = observer_obs.board

                        true_hand = session.game.player_hands[actor]
                        prior = beliefs[viewer].copy()
                        log_probs = model.predict_log_probs(observer_obs, viewer, true_hand).squeeze(0)
                        correct_actions += int(log_probs.argmax().item() == int(action))
                        total_actions += 1

                        total_prior_true_hand_prob += model.belief_true_hand_probability(prior, true_hand)
                        total_prior_top1 += model.belief_top1_correct(prior, true_hand)

                        beliefs[viewer] = model.update_belief(prior, observer_obs, viewer, action)
                        total_true_hand_prob += model.belief_true_hand_probability(beliefs[viewer], true_hand)
                        total_belief_top1 += model.belief_top1_correct(beliefs[viewer], true_hand)
                        total_tvd_shift += model.tvd(prior, beliefs[viewer])

                    session.step(action)

    denom = max(total_actions, 1)
    return {
        "action_accuracy": round(correct_actions / denom, 4),
        "belief_top1_accuracy": round(total_belief_top1 / denom, 4),
        "prior_top1_accuracy": round(total_prior_top1 / denom, 4),
        "mean_true_hand_posterior": round(total_true_hand_prob / denom, 4),
        "mean_true_hand_prior": round(total_prior_true_hand_prob / denom, 4),
        "mean_tvd_shift": round(total_tvd_shift / denom, 4),
        "num_actions": total_actions,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate the hand-conditioned action model.")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=ROOT / "outputs" / EXPERIMENT_ID / "checkpoint.pt",
        help="Experiment checkpoint path.",
    )
    parser.add_argument("--probe-agent", type=str, default="modulated_value", help="Observer/probe agent.")
    parser.add_argument("--opponents", type=str, default=",".join(DEFAULT_EVAL_POOL), help="Comma-separated target opponents.")
    parser.add_argument("--sessions", type=int, default=20, help="Sessions per seat configuration.")
    parser.add_argument("--hands-per-session", type=int, default=20, help="Hands per session.")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs" / EXPERIMENT_ID / "evaluation.json",
        help="Evaluation JSON path.",
    )
    args = parser.parse_args()

    opponents = tuple(item.strip() for item in args.opponents.split(",") if item.strip())
    model = HandConditionedActionModel(model_path=str(args.checkpoint))
    model.set_train_mode(False)

    per_opponent = {
        opponent_id: evaluate_target(model, opponent_id, args.probe_agent, args.sessions, args.hands_per_session)
        for opponent_id in opponents
    }

    mean_action_accuracy = sum(v["action_accuracy"] for v in per_opponent.values()) / len(per_opponent)
    mean_belief_top1 = sum(v["belief_top1_accuracy"] for v in per_opponent.values()) / len(per_opponent)
    mean_prior_top1 = sum(v["prior_top1_accuracy"] for v in per_opponent.values()) / len(per_opponent)
    mean_true_hand_posterior = sum(v["mean_true_hand_posterior"] for v in per_opponent.values()) / len(per_opponent)
    mean_true_hand_prior = sum(v["mean_true_hand_prior"] for v in per_opponent.values()) / len(per_opponent)
    mean_tvd_shift = sum(v["mean_tvd_shift"] for v in per_opponent.values()) / len(per_opponent)

    payload = {
        "experiment_id": EXPERIMENT_ID,
        "checkpoint": str(args.checkpoint),
        "probe_agent": args.probe_agent,
        "metrics": {
            "mean_action_accuracy": round(mean_action_accuracy, 4),
            "mean_belief_top1_accuracy": round(mean_belief_top1, 4),
            "mean_prior_top1_accuracy": round(mean_prior_top1, 4),
            "belief_top1_lift": round(mean_belief_top1 - mean_prior_top1, 4),
            "mean_true_hand_posterior": round(mean_true_hand_posterior, 4),
            "mean_true_hand_prior": round(mean_true_hand_prior, 4),
            "true_hand_probability_lift": round(mean_true_hand_posterior - mean_true_hand_prior, 4),
            "mean_tvd_shift": round(mean_tvd_shift, 4),
        },
        "per_opponent": per_opponent,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
