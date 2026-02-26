"""
Pruned History Trainer — session-based training for the pruned history agent.

Extends AdaptiveTrainer to carry BOTH opponent_stats AND action_history
into simulated post-action states used for TD learning.

AdaptiveTrainer already handles:
  - Session-based training (PokerSession with multi-hand stats accumulation)
  - opponent_stats carry-forward in collect_episode()
  - Session-based train() loop

This trainer adds:
  - action_history carry-forward (same as AdaptiveHistoryTrainer)
  - Enhanced debug_episode() with both stats and history info
"""

import os
import torch
from dataclasses import replace
from typing import Dict, List, Tuple

from src.engine.leduc_game import LeducGame, Action
from src.engine.poker_session import PokerSession
from src.agents.base import BaseAgent
from src.training.adaptive_trainer import AdaptiveTrainer


# Mapping from Action enum to history string
_ACTION_TO_NAME = {
    Action.FOLD: "FOLD",
    Action.CALL: "CALL",
    Action.RAISE: "RAISE",
}


class PrunedHistoryTrainer(AdaptiveTrainer):
    """
    Session-based trainer for PrunedHistoryAgent.

    Combines AdaptiveTrainer's session-based stats accumulation with
    action history carry-forward in collect_episode().
    """

    def __init__(self, agent: BaseAgent, learning_rate: float = 1e-4,
                 hands_per_session: int = 30):
        super().__init__(agent, learning_rate=learning_rate,
                         hands_per_session=hands_per_session)

    def collect_episode(self) -> List[Tuple[List[List[torch.Tensor]], List[float]]]:
        """Play one session of hands, carrying both stats and history into post-states.

        Returns a list of (chains, rewards) tuples -- one per hand in the session.
        Stats accumulate across hands within the session and reset between sessions.
        """
        self.session.reset()
        session_data = []

        for _ in range(self.hands_per_session):
            self.session.new_hand()
            chains = [[], []]

            while not self.session.is_finished:
                current_player = self.session.current_player
                obs = self.session.get_observation(viewer_id=current_player)

                action = self.agent.select_action(obs)
                if isinstance(action, tuple):
                    action = action[0]

                # Record post-action state with BOTH stats and history
                post_obs, _ = LeducGame.simulate_action(obs, action)

                # Carry opponent_stats
                if obs.opponent_stats is not None:
                    post_obs = replace(post_obs, opponent_stats=obs.opponent_stats)

                # Carry extended action_history
                action_name = _ACTION_TO_NAME[action]
                current_history = obs.action_history if obs.action_history else ()
                extended_history = current_history + ((current_player, action_name),)
                post_obs = replace(post_obs, action_history=extended_history)

                encoded = self.agent.encode_observation(post_obs, viewer_id=current_player)
                chains[current_player].append(encoded)

                self.session.step(action)

            rewards = self.session.game.get_reward()
            session_data.append((chains, rewards))

        return session_data

    def debug_episode(self) -> Dict:
        """Play one session and return a debug trace with both stats and history visible."""
        self.session.reset()
        episode_trace = []

        old_train_mode = self.agent.train_mode
        self.agent.set_train_mode(False)

        # Play a short session to build up some stats, then trace the last hand
        warmup_hands = min(5, self.hands_per_session - 1)

        for _ in range(warmup_hands):
            self.session.new_hand()
            while not self.session.is_finished:
                player = self.session.current_player
                obs = self.session.get_observation(viewer_id=player)
                action = self.agent.select_action(obs)
                self.session.step(action)

        # Now trace the final hand with full detail
        self.session.new_hand()
        while not self.session.is_finished:
            current_player = self.session.current_player
            obs = self.session.get_observation(viewer_id=current_player)
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
                "opponent_stats": (obs.opponent_stats.to_feature_vector()
                                   if obs.opponent_stats else None),
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
            self.session.step(action)

        rewards = self.session.game.get_reward()

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
            "session_analytics": self.session.get_analytics(),
        }

    def update_params(self, params: Dict):
        """Updates trainer parameters."""
        if "lr" in params:
            new_lr = params["lr"]
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = new_lr
            print(f"Learning rate updated to: {new_lr}")
        if "hands_per_session" in params:
            self.hands_per_session = params["hands_per_session"]
            print(f"Hands per session updated to: {self.hands_per_session}")
