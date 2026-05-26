"""REINFORCE trainer for repr_policy_v1 experiment.

Trains any of the three agent variants (VanillaPolicyAgent, ReprPolicyAgent,
ReprPolicyFineTuneAgent) using the REINFORCE algorithm with self-play.

Both player positions are played by the same agent. Log-probs are tracked
separately per position; each position's log-probs are paired with that
position's final reward.
"""

import torch
import torch.optim as optim
from typing import Dict, List

from preliminary_experiments.promoted_registry.base_trainer import BaseTrainer
from engine.leduc_game import LeducGame, Action


class REINFORCETrainer(BaseTrainer):
    """REINFORCE trainer for repr_policy_v1 agents.

    Episode collection: full Leduc game in self-play, log-probs tracked per seat.
    Batch update: REINFORCE loss = -mean(log_prob * reward) over all (seat, episode) pairs,
    with optional reward normalization.
    """

    def __init__(self, agent, learning_rate: float = 1e-4,
                 normalize_rewards: bool = True):
        super().__init__(agent, eval_interval=500, eval_num_games=200)
        self.normalize_rewards = normalize_rewards
        self.game = LeducGame()

        # Build optimizer over all trainable parameters
        self.optimizer = optim.Adam(agent.parameters(), lr=learning_rate)

    def collect_episode(self) -> dict:
        """Play one full LeducGame in self-play.

        Returns:
            dict with:
              - log_probs: list[list[Tensor]], log_probs[seat] = list of log_prob tensors
              - rewards: list[float], final chip reward for each seat
        """
        self.game.reset()
        log_probs = [[], []]  # separate tracking per seat

        self.agent.set_train_mode(True)

        while not self.game.is_finished:
            player = self.game.current_player
            obs = self.game.get_observation(viewer_id=player)

            action, log_prob = self.agent.select_action_with_log_prob(obs, viewer_id=player)
            log_probs[player].append(log_prob)

            self.game.step(action)

        rewards = self.game.get_reward()  # [r0, r1]
        return {'log_probs': log_probs, 'rewards': rewards}

    def update_model(self, batch_data: list) -> float:
        """REINFORCE update over a batch of episodes.

        For each episode and each seat:
            loss += -log_prob * reward   (summed over all actions in that seat)

        Reward normalization: subtract batch mean, divide by batch std+eps.
        """
        self.optimizer.zero_grad()
        total_loss = torch.tensor(0.0)

        # Collect all (log_prob_sum, reward) pairs for optional normalization
        pairs = []
        for episode in batch_data:
            log_probs = episode['log_probs']
            rewards = episode['rewards']
            for seat in [0, 1]:
                if not log_probs[seat]:
                    continue
                lp_sum = torch.stack(log_probs[seat]).sum()
                reward = float(rewards[seat])
                pairs.append((lp_sum, reward))

        if not pairs:
            return 0.0

        rewards_tensor = torch.tensor([r for _, r in pairs])

        if self.normalize_rewards and len(pairs) > 1:
            mean_r = rewards_tensor.mean()
            std_r = rewards_tensor.std() + 1e-8
            rewards_normalized = (rewards_tensor - mean_r) / std_r
        else:
            rewards_normalized = rewards_tensor

        for i, (lp_sum, _) in enumerate(pairs):
            total_loss = total_loss - lp_sum * rewards_normalized[i]

        total_loss = total_loss / len(pairs)
        total_loss.backward()
        self.optimizer.step()

        return total_loss.item()

    def update_params(self, params: Dict):
        """Allow dynamic learning rate adjustment."""
        if 'lr' in params:
            for pg in self.optimizer.param_groups:
                pg['lr'] = params['lr']
            print(f"Learning rate updated to: {params['lr']}")
