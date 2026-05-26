"""
Belief-CFR Trainer -- Self-play trainer for the Belief-CFR Agent.

Unlike the standard BeliefTrainer, this trainer does NOT train a
likelihood model (the CFR Nash strategy is frozen). It only trains
the value network via TD(0) on post-action state chains.

Training protocol:
  1. Self-play using the belief-CFR agent against itself.
  2. TD(0) on post-action state chains.
  3. Belief updates use the frozen CFR strategy store (no learning).
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, List, Tuple
from src.engine.leduc_game import LeducGame, Action
from src.engine.observation import Observation
from src.agents.belief_cfr import BeliefCfrAgent
from src.training.base import BaseTrainer


class BeliefCfrTrainer(BaseTrainer):
    """
    Trainer for BeliefCfrAgent.

    Only trains the value network via TD(0). The CFR-based likelihood
    model is frozen and never updated.
    """

    def __init__(self, agent: BeliefCfrAgent, learning_rate: float = 1e-4):
        super().__init__(agent, eval_interval=50, eval_num_games=100)

        # Value network optimizer (only thing we train)
        self.value_optimizer = optim.Adam(
            self.agent.model.parameters(), lr=learning_rate
        )
        self.value_criterion = nn.MSELoss()

        self.game = LeducGame()

    def collect_episode(self) -> Tuple[List[List[torch.Tensor]], List[float]]:
        """
        Play one episode of self-play.

        Returns:
            chains: per-player post-action state chains for TD(0)
            rewards: final rewards for each player
        """
        self.game.reset()
        chains = [[], []]  # chains[p] = P's post-action encoded states

        while not self.game.is_finished:
            current_player = self.game.current_player
            obs = self.game.get_observation(viewer_id=current_player)

            # Get action from belief-CFR agent
            action = self.agent.select_action(obs)
            if isinstance(action, tuple):
                action = action[0]

            # Compute belief at this point for encoding
            belief = self.agent.compute_belief_from_history(obs)

            # Record post-action state for TD(0)
            post_obs, _ = LeducGame.simulate_action(obs, action)
            encoded = self.agent.encode_observation(
                post_obs, viewer_id=current_player, belief=belief
            )
            chains[current_player].append(encoded)

            self.game.step(action)

        rewards = self.game.get_reward()
        return chains, rewards

    def update_model(self, batch_data: list) -> float:
        """
        Update the value network via TD(0).

        Args:
            batch_data: List of (chains, rewards) tuples.

        Returns:
            Loss value.
        """
        self.value_optimizer.zero_grad()
        total_losses = []

        for chains, rewards in batch_data:
            for p_idx in [0, 1]:
                chain = chains[p_idx]
                if not chain:
                    continue

                for t in range(len(chain)):
                    prediction = self.agent.model(chain[t]).squeeze(0)

                    if t == len(chain) - 1:
                        target = torch.FloatTensor([rewards[p_idx]])
                    else:
                        with torch.no_grad():
                            target = self.agent.model(chain[t + 1]).squeeze(0)

                    loss = self.value_criterion(prediction, target)
                    total_losses.append(loss)

        if total_losses:
            mean_loss = torch.stack(total_losses).mean()
            mean_loss.backward()
            self.value_optimizer.step()
            return mean_loss.item()
        return 0.0

    def debug_episode(self) -> Dict:
        """Play one episode and record detailed trace with belief vectors."""
        self.game.reset()
        episode_trace = []

        old_train_mode = self.agent.train_mode
        self.agent.set_train_mode(False)

        while not self.game.is_finished:
            current_player = self.game.current_player
            obs = self.game.get_observation(viewer_id=current_player)
            belief = self.agent.compute_belief_from_history(obs)

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
                },
                "belief": belief.tolist(),
                "evaluations": [
                    {
                        "action": e["action"].name,
                        "value": e["value"],
                        "belief": e["belief"].tolist(),
                    } for e in evaluations
                ],
                "selected_action": action.name,
            }
            episode_trace.append(step_info)
            self.game.step(action)

        rewards = self.game.get_reward()

        for step in episode_trace:
            player_reward = rewards[step["player_id"]]
            step["true_value"] = player_reward

        self.agent.set_train_mode(old_train_mode)
        return {
            "trace": episode_trace,
            "final_rewards": rewards,
            "eval_type": "belief_cfr",
            "player_hands": list(self.game.player_hands),
        }

    def update_params(self, params: Dict):
        """Update learning rates."""
        if "lr" in params:
            for pg in self.value_optimizer.param_groups:
                pg['lr'] = params["lr"]
