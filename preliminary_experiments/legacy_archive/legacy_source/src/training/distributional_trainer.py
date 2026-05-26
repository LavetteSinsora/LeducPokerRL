"""
Distributional Trainer for the DistributionalValueAgent.

Uses two separate loss functions:
  - Value network: MSE loss with TD(0) targets (same as SelfPlayTrainer)
  - Variance network: MSE loss predicting (reward - mean)^2

The two networks have separate optimizers to prevent gradient interference.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Tuple

from src.engine.leduc_game import LeducGame, Action
from src.agents.base import BaseAgent
from src.training.base import BaseTrainer


class DistributionalTrainer(BaseTrainer):
    """
    Self-play trainer with separate value and variance learning.
    """

    def __init__(self, agent: BaseAgent, learning_rate: float = 1e-4,
                 var_weight: float = 0.5, kappa: float = 1.0):
        super().__init__(agent, eval_interval=50, eval_num_games=100)
        # Separate optimizers for independent gradient updates
        self.value_optimizer = optim.Adam(
            self.agent.model.value_net.parameters(), lr=learning_rate)
        self.var_optimizer = optim.Adam(
            self.agent.model.var_net.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()
        self.game = LeducGame()

        # Keep for API compatibility
        if hasattr(self.agent, 'taus'):
            self.taus = self.agent.taus

    def collect_episode(self) -> Tuple[List[List[torch.Tensor]], List[float]]:
        """Play one self-play episode."""
        self.game.reset()
        chains = [[], []]

        while not self.game.is_finished:
            current_player = self.game.current_player
            obs = self.game.get_observation(viewer_id=current_player)
            action = self.agent.select_action(obs)
            if isinstance(action, tuple):
                action = action[0]

            post_obs, _ = LeducGame.simulate_action(obs, action)
            encoded = self.agent.encode_observation(post_obs, viewer_id=current_player)
            chains[current_player].append(encoded)

            self.game.step(action)

        rewards = self.game.get_reward()
        return chains, rewards

    def update_model(self, batch_data: list) -> float:
        """
        TD(0) for value network, squared-error for variance network.
        Separate backward passes to prevent gradient interference.
        """
        # ---- Phase 1: Value network update (identical to SelfPlayTrainer) ----
        self.value_optimizer.zero_grad()
        value_losses = []

        for chains, rewards in batch_data:
            for p_idx in [0, 1]:
                chain = chains[p_idx]
                if not chain:
                    continue
                for t in range(len(chain)):
                    prediction = self.agent.model.value_net(chain[t]).squeeze(0)
                    if t == len(chain) - 1:
                        target = torch.FloatTensor([rewards[p_idx]])
                    else:
                        with torch.no_grad():
                            target = self.agent.model.value_net(chain[t + 1]).squeeze(0)
                    loss = self.criterion(prediction, target)
                    value_losses.append(loss)

        value_loss_val = 0.0
        if value_losses:
            mean_value_loss = torch.stack(value_losses).mean()
            mean_value_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.agent.model.value_net.parameters(), max_norm=1.0)
            self.value_optimizer.step()
            value_loss_val = mean_value_loss.item()

        # ---- Phase 2: Variance network update ----
        self.var_optimizer.zero_grad()
        var_losses = []

        for chains, rewards in batch_data:
            for p_idx in [0, 1]:
                chain = chains[p_idx]
                if not chain:
                    continue
                for t in range(len(chain)):
                    pred_var = self.agent.model.var_net(chain[t]).squeeze(0)

                    with torch.no_grad():
                        pred_mean = self.agent.model.value_net(chain[t]).squeeze(0)

                    if t == len(chain) - 1:
                        # Terminal: variance target = (reward - mean)^2
                        target_var = torch.FloatTensor([(rewards[p_idx] - pred_mean.item()) ** 2])
                    else:
                        with torch.no_grad():
                            next_var = self.agent.model.var_net(chain[t + 1]).squeeze(0)
                            next_mean = self.agent.model.value_net(chain[t + 1]).squeeze(0)
                        # Propagate: Var = next_var + (next_mean - curr_mean)^2
                        target_var = next_var + (next_mean - pred_mean) ** 2

                    loss = self.criterion(pred_var, target_var)
                    var_losses.append(loss)

        var_loss_val = 0.0
        if var_losses:
            mean_var_loss = torch.stack(var_losses).mean()
            mean_var_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.agent.model.var_net.parameters(), max_norm=1.0)
            self.var_optimizer.step()
            var_loss_val = mean_var_loss.item()

        return value_loss_val + var_loss_val

    def debug_episode(self) -> Dict:
        """Plays one episode and records detailed mean-variance evaluations."""
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
                },
                "evaluations": [
                    {
                        "action": e["action"].name,
                        "value": e["value"],
                        "mean": e["mean"],
                        "std": e["std"],
                        "action_id": e["action"].value,
                    } for e in evaluations
                ],
                "selected_action": action.name,
                "selected_action_id": action.value,
            }

            episode_trace.append(step_info)
            self.game.step(action)

        rewards = self.game.get_reward()

        for step in episode_trace:
            player_reward = rewards[step["player_id"]]
            step["true_value"] = player_reward
            pred_val = next(e["mean"] for e in step["evaluations"]
                           if e["action"] == step["selected_action"])
            step["prediction_error"] = (pred_val - player_reward) ** 2

        self.agent.set_train_mode(old_train_mode)
        return {
            "trace": episode_trace,
            "final_rewards": rewards,
            "eval_type": "distributional",
        }

    def update_params(self, params: Dict):
        """Updates the learning rate of both optimizers."""
        if "lr" in params:
            new_lr = params["lr"]
            for opt in [self.value_optimizer, self.var_optimizer]:
                for param_group in opt.param_groups:
                    param_group['lr'] = new_lr
            print(f"Learning rate updated to: {new_lr}")
