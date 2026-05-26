"""ValueDimTrainer: TD(0) self-play trainer for ValueDimAgent.

Identical training recipe to value_based/trainer.py — same episode collection,
same TD(0) update, same loss. Only the agent's network size differs.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Tuple

from preliminary_experiments.promoted_registry.base_trainer import BaseTrainer
from engine.leduc_game import LeducGame, Action

from experiments.representation_learning.value_dim_search_v1.agent import ValueDimAgent


class ValueDimTrainer(BaseTrainer):
    """
    Self-play TD(0) trainer for ValueDimAgent.

    Identical to SelfPlayTrainer in agents/value_based/trainer.py.
    Both players share the same agent instance.
    """

    def __init__(self, agent: ValueDimAgent, learning_rate: float = 1e-4):
        super().__init__(agent, eval_interval=500, eval_num_games=200)
        self.optimizer = optim.Adam(self.agent.model.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()
        self.game = LeducGame()

    def collect_episode(self) -> Tuple[List[List[torch.Tensor]], List[float]]:
        """Play one episode, returning per-player post-action state chains and rewards."""
        self.game.reset()
        chains = [[], []]  # chains[0] = P0's post-action states, chains[1] = P1's

        while not self.game.is_finished:
            current_player = self.game.current_player
            obs = self.game.get_observation(viewer_id=current_player)
            action = self.agent.select_action(obs)
            if isinstance(action, tuple):
                action = action[0]

            # Record post-action state for the acting player
            post_obs, _ = LeducGame.simulate_action(obs, action)
            encoded = self.agent.encode_observation(post_obs, viewer_id=current_player)
            chains[current_player].append(encoded)

            self.game.step(action)

        rewards = self.game.get_reward()
        return chains, rewards

    def update_model(self, batch_data: list) -> float:
        """TD(0) on per-player post-action state chains.
        Each chain: V(s_t) -> V(s_{t+1}), last state -> terminal reward."""
        self.optimizer.zero_grad()
        total_losses = []

        for chains, rewards in batch_data:
            for p_idx in [0, 1]:
                chain = chains[p_idx]
                if not chain:
                    continue

                for t in range(len(chain)):
                    prediction = self.agent.model(chain[t]).squeeze(0)

                    if t == len(chain) - 1:
                        # Last action -> target is terminal reward
                        target = torch.FloatTensor([rewards[p_idx]])
                    else:
                        # Bootstrap from next state in this player's chain
                        with torch.no_grad():
                            target = self.agent.model(chain[t + 1]).squeeze(0)

                    loss = self.criterion(prediction, target)
                    total_losses.append(loss)

        if total_losses:
            mean_loss = torch.stack(total_losses).mean()
            mean_loss.backward()
            self.optimizer.step()
            return mean_loss.item()
        return 0.0

    def update_params(self, params: Dict):
        """Updates the learning rate of the optimizer."""
        if "lr" in params:
            new_lr = params["lr"]
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = new_lr
            print(f"Learning rate updated to: {new_lr}")
