"""Generate sample belief traces for the hand-conditioned action model."""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine.poker_session import PokerSession
from experiments.hand_conditioned_action_model_v1.agent import (
    CARDS,
    HandConditionedActionModel,
    initialize_belief,
    update_belief_with_board,
)
from experiments.hand_conditioned_action_model_v1.trainer import load_registry_agent


EXPERIMENT_ID = "hand_conditioned_action_model_v1"


def collect_traces(model, target_id: str, probe_id: str, num_hands: int) -> dict:
    target_agent = load_registry_agent(target_id)
    probe_agent = load_registry_agent(probe_id)
    session = PokerSession()
    traces = []

    while len(traces) < num_hands:
        session.new_hand()
        beliefs = [
            initialize_belief(session.game.player_hands[0]),
            initialize_belief(session.game.player_hands[1]),
        ]
        board_seen = [None, None]
        hand_trace = {"target": target_id, "steps": []}

        while not session.is_finished:
            actor = session.current_player
            viewer = 1 - actor
            agents = [target_agent, probe_agent]
            actor_obs = session.get_observation(viewer_id=actor)
            observer_obs = session.get_observation(viewer_id=viewer)
            action = agents[actor].select_action(actor_obs)
            if isinstance(action, tuple):
                action = action[0]

            if actor == 0:
                if observer_obs.board is not None and board_seen[viewer] is None:
                    beliefs[viewer] = update_belief_with_board(
                        beliefs[viewer],
                        viewer_hand=observer_obs.player_hand,
                        board=observer_obs.board,
                    )
                    board_seen[viewer] = observer_obs.board

                prior = beliefs[viewer].copy()
                posterior = model.update_belief(prior, observer_obs, viewer, action)
                true_hand = session.game.player_hands[actor]

                hand_trace["steps"].append(
                    {
                        "board": observer_obs.board,
                        "pot": list(observer_obs.pot),
                        "current_round": observer_obs.current_round,
                        "raises_this_round": observer_obs.raises_this_round,
                        "stats": model.stats_to_tensor(observer_obs).tolist(),
                        "action": action.name,
                        "true_hand": true_hand,
                        "prior": prior.tolist(),
                        "posterior": posterior.tolist(),
                        "per_hand_action_probs": {
                            hand: model.predict_action_probs(observer_obs, viewer, hand).squeeze(0).tolist()
                            for hand in CARDS
                        },
                    }
                )
                beliefs[viewer] = posterior

            session.step(action)

        if hand_trace["steps"]:
            traces.append(hand_trace)

    return {"target": target_id, "traces": traces}


def main():
    parser = argparse.ArgumentParser(description="Diagnose the hand-conditioned action model.")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=ROOT / "outputs" / EXPERIMENT_ID / "checkpoint.pt",
        help="Experiment checkpoint path.",
    )
    parser.add_argument("--targets", type=str, default="heuristic,adaptive_value,cfr", help="Comma-separated targets.")
    parser.add_argument("--probe-agent", type=str, default="modulated_value", help="Probe agent.")
    parser.add_argument("--hands", type=int, default=3, help="Number of example hands per target.")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs" / EXPERIMENT_ID / "diagnostics.json",
        help="Diagnostic JSON path.",
    )
    args = parser.parse_args()

    targets = tuple(item.strip() for item in args.targets.split(",") if item.strip())
    model = HandConditionedActionModel(model_path=str(args.checkpoint))
    model.set_train_mode(False)

    payload = {
        "experiment_id": EXPERIMENT_ID,
        "checkpoint": str(args.checkpoint),
        "probe_agent": args.probe_agent,
        "diagnostics": [collect_traces(model, target_id, args.probe_agent, args.hands) for target_id in targets],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
