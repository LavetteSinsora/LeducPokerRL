"""
History Value Trainer — TD(0) self-play for the HistoryValueAgent.

Extends SelfPlayTrainer to ensure action_history is correctly propagated
into simulated post-action states used for TD learning.

The game engine already populates action_history in get_observation(),
but simulate_action() does not carry it forward. This trainer manually
extends the history when building the post-action encoded states.

update_model() is inherited unchanged — same TD(0) algorithm.
"""

import torch
from dataclasses import replace
from typing import Dict, List, Tuple

from src.engine.leduc_game import LeducGame, Action
from src.agents.base import BaseAgent
from src.training.value_based_trainer import SelfPlayTrainer


# Mapping from Action enum to history string
_ACTION_TO_NAME = {
    Action.FOLD: "FOLD",
    Action.CALL: "CALL",
    Action.RAISE: "RAISE",
}


class HistoryValueTrainer(SelfPlayTrainer):
    """
    Trainer for HistoryValueAgent.

    Same TD(0) self-play as SelfPlayTrainer, but correctly propagates
    action_history into simulated post-action observations so the
    history features in the encoded state are meaningful.
    """

    def __init__(self, agent: BaseAgent, learning_rate: float = 1e-4):
        super().__init__(agent, learning_rate=learning_rate)

    def collect_episode(self) -> Tuple[List[List[torch.Tensor]], List[float]]:
        """Play one episode, returning per-player post-action state chains and rewards.

        Same structure as SelfPlayTrainer.collect_episode(), but carries
        action_history into simulated post-action observations.
        """
        self.game.reset()
        chains = [[], []]

        while not self.game.is_finished:
            current_player = self.game.current_player
            obs = self.game.get_observation(viewer_id=current_player)
            action = self.agent.select_action(obs)
            if isinstance(action, tuple):
                action = action[0]

            # Simulate the post-action state (with board masking)
            post_obs, _ = LeducGame.simulate_action(obs, action)

            # Extend action_history for the simulated state
            action_name = _ACTION_TO_NAME[action]
            current_history = obs.action_history if obs.action_history else ()
            extended_history = current_history + ((current_player, action_name),)
            post_obs = replace(post_obs, action_history=extended_history)

            encoded = self.agent.encode_observation(post_obs, viewer_id=current_player)
            chains[current_player].append(encoded)

            self.game.step(action)

        rewards = self.game.get_reward()
        return chains, rewards

    def debug_episode(self) -> Dict:
        """Play one episode in debug mode with action history visible."""
        self.game.reset()
        episode_trace = []

        old_train_mode = self.agent.train_mode
        self.agent.set_train_mode(False)

        while not self.game.is_finished:
            current_player = self.game.current_player
            obs = self.game.get_observation(viewer_id=current_player)
            evaluations = self.agent.get_action_evaluations(obs)

            selected_eval = max(evaluations, key=lambda x: x["value"])
            action = selected_eval["action"]

            step_info = {
                "player_id": current_player,
                "observation": {
                    "player_hand": obs.player_hand,
                    "board": obs.board,
                    "pot": obs.pot,
                    "current_round": obs.current_round,
                    "action_history": list(obs.action_history) if obs.action_history else [],
                },
                "evaluations": [
                    {
                        "action": e["action"].name,
                        "value": e["value"],
                        "action_id": e["action"].value,
                        "encoded_state": e["encoded"].squeeze(0).tolist(),
                    } for e in evaluations
                ],
                "selected_action": action.name,
                "selected_action_id": action.value,
                "encoded_state": selected_eval["encoded"].squeeze(0).tolist(),
            }
            episode_trace.append(step_info)
            self.game.step(action)

        rewards = self.game.get_reward()

        for step in episode_trace:
            player_reward = rewards[step["player_id"]]
            step["true_value"] = player_reward
            pred_val = next(e["value"] for e in step["evaluations"]
                           if e["action"] == step["selected_action"])
            step["prediction_error"] = (pred_val - player_reward) ** 2

        self.agent.set_train_mode(old_train_mode)
        return {
            "trace": episode_trace,
            "final_rewards": rewards,
            "eval_type": "value",
        }

    def update_params(self, params: Dict):
        """Updates trainer parameters (e.g., learning rate)."""
        if "lr" in params:
            new_lr = params["lr"]
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = new_lr
            print(f"Learning rate updated to: {new_lr}")
