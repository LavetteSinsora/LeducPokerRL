"""Contrastive trainer for representation learning.

Collects data via self-play with a trained value-based agent,
stores (state_encoding, terminal_reward, episode_id) tuples in a
replay buffer, and trains the contrastive encoder using one of
three loss formulations (L0/L1/L2).
"""

import random
from collections import deque
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim

from preliminary_experiments.promoted_registry.base_trainer import BaseTrainer
from agents.value_based.agent import ValueBasedAgent
from engine.leduc_game import LeducGame, Action

from experiments.representation_learning.contrastive_repr_v1.agent import ContrastiveReprAgent
from experiments.representation_learning.contrastive_repr_v1.losses import (
    calibrate_beta,
    rank_n_contrast_loss_vectorized,
    soft_distance_correlation_loss,
    vicreg_variance_loss,
)


class ReplayBuffer:
    """Stores (state_encoding, reward, episode_id) tuples."""

    def __init__(self, capacity: int = 5000):
        self.capacity = capacity
        self.buffer: deque = deque(maxlen=capacity)

    def add(self, state: torch.Tensor, reward: float, episode_id: int):
        self.buffer.append((state.detach(), reward, episode_id))

    def sample(self, n: int, cross_trajectory_only: bool = False
               ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Sample n items from the buffer.

        Args:
            n: number of samples
            cross_trajectory_only: if True, return episode_ids so the
                caller can filter same-episode pairs

        Returns:
            (states [N, 15], rewards [N], episode_ids [N] or None)
        """
        n = min(n, len(self.buffer))
        items = random.sample(list(self.buffer), n)

        states = torch.cat([s for s, _, _ in items], dim=0)   # (N, 15)
        rewards = torch.tensor([r for _, r, _ in items], dtype=torch.float)  # (N,)
        if cross_trajectory_only:
            ep_ids = torch.tensor([e for _, _, e in items])
            return states, rewards, ep_ids
        return states, rewards, None

    def __len__(self):
        return len(self.buffer)


class ContrastiveTrainer(BaseTrainer):
    """Trains a ContrastiveReprAgent using contrastive/TD losses.

    Args:
        agent: the contrastive encoder agent
        data_agent_path: path to trained value-based checkpoint for data collection
        loss_type: 'L0' (TD control), 'L1' (distance correlation), 'L2' (Rank-N-Contrast)
        learning_rate: Adam LR
        contrastive_batch_size: samples drawn from replay buffer per update
        buffer_capacity: replay buffer size
        lambda_var: VICReg variance weight (L1 only)
        temperature: RnC temperature (L2 only)
        beta: distance exchange rate (L1 only); None = auto-calibrate
        cross_trajectory_only: exclude same-episode pairs from contrastive loss
    """

    def __init__(self, agent: ContrastiveReprAgent,
                 data_agent_path: str = 'agents/value_based/checkpoint.pt',
                 loss_type: str = 'L1',
                 learning_rate: float = 1e-4,
                 contrastive_batch_size: int = 256,
                 buffer_capacity: int = 5000,
                 lambda_var: float = 0.1,
                 temperature: float = 0.5,
                 beta: Optional[float] = None,
                 cross_trajectory_only: bool = False):

        super().__init__(agent, eval_interval=500, eval_num_games=100)

        self.loss_type = loss_type.upper()
        assert self.loss_type in ('L0', 'L1', 'L2'), f"Unknown loss type: {loss_type}"

        self.contrastive_batch_size = contrastive_batch_size
        self.lambda_var = lambda_var
        self.temperature = temperature
        self.beta = beta
        self.beta_calibrated = beta is not None
        self.cross_trajectory_only = cross_trajectory_only

        # Data collection agent (frozen value-based policy)
        self.data_agent = ValueBasedAgent(model_path=data_agent_path)
        self.data_agent.set_train_mode(False)

        self.game = LeducGame()
        self.replay_buffer = ReplayBuffer(capacity=buffer_capacity)
        self.episode_counter = 0

        # Optimizer over encoder (and value head for L0)
        params = list(agent.encoder.parameters())
        if self.loss_type == 'L0' and agent.value_head is not None:
            params += list(agent.value_head.parameters())
        self.optimizer = optim.Adam(params, lr=learning_rate)
        self.criterion = nn.MSELoss()

        # Logging
        self.loss_history: List[float] = []

    # ------------------------------------------------------------------
    # Data collection
    # ------------------------------------------------------------------

    def collect_episode(self) -> Tuple[List[List[torch.Tensor]], List[float]]:
        """Play one self-play episode using the data agent.

        Returns:
            (chains, rewards) where chains[p] is a list of encoded state
            tensors for player p, and rewards is [R0, R1].
        """
        self.game.reset()
        chains = [[], []]

        while not self.game.is_finished:
            current_player = self.game.current_player
            obs = self.game.get_observation(viewer_id=current_player)
            action = self.data_agent.select_action(obs)

            # Record the pre-action observation encoded by OUR encoder
            encoded = self.agent.encode_observation(obs, viewer_id=current_player)
            chains[current_player].append(encoded)

            self.game.step(action)

        rewards = self.game.get_reward()
        return chains, rewards

    def _collect_and_store(self, num_episodes: int = 8):
        """Collect episodes and add to replay buffer."""
        for _ in range(num_episodes):
            chains, rewards = self.collect_episode()
            self.episode_counter += 1

            for p_idx in (0, 1):
                for state_enc in chains[p_idx]:
                    self.replay_buffer.add(
                        state_enc, rewards[p_idx], self.episode_counter
                    )

    # ------------------------------------------------------------------
    # Loss computation
    # ------------------------------------------------------------------

    def _compute_l0_loss(self, batch_data: list) -> float:
        """TD(0) loss using encoder + value head (control baseline)."""
        self.optimizer.zero_grad()
        total_losses = []

        for chains, rewards in batch_data:
            for p_idx in (0, 1):
                chain = chains[p_idx]
                if not chain:
                    continue

                for t in range(len(chain)):
                    z = self.agent.encoder(chain[t])
                    prediction = self.agent.value_head(z).squeeze(0)

                    if t == len(chain) - 1:
                        target = torch.FloatTensor([rewards[p_idx]])
                    else:
                        with torch.no_grad():
                            z_next = self.agent.encoder(chain[t + 1])
                            target = self.agent.value_head(z_next).squeeze(0)

                    loss = self.criterion(prediction, target)
                    total_losses.append(loss)

        if total_losses:
            mean_loss = torch.stack(total_losses).mean()
            mean_loss.backward()
            self.optimizer.step()
            return mean_loss.item()
        return 0.0

    def _compute_contrastive_loss(self) -> float:
        """Contrastive loss (L1 or L2) from replay buffer samples."""
        if len(self.replay_buffer) < self.contrastive_batch_size:
            return 0.0

        states, rewards, ep_ids = self.replay_buffer.sample(
            self.contrastive_batch_size,
            cross_trajectory_only=self.cross_trajectory_only
        )

        # Forward through encoder
        z = self.agent.encoder(states)

        # Auto-calibrate beta on first real batch
        if self.loss_type == 'L1' and not self.beta_calibrated:
            self.beta = calibrate_beta(z, rewards)
            self.beta_calibrated = True
            print(f"Auto-calibrated beta = {self.beta:.4f}")

        # Compute contrastive loss
        if self.loss_type == 'L1':
            loss = soft_distance_correlation_loss(z, rewards, beta=self.beta)
            loss = loss + self.lambda_var * vicreg_variance_loss(z)
        else:  # L2
            loss = rank_n_contrast_loss_vectorized(z, rewards,
                                                   temperature=self.temperature)

        if loss.requires_grad:
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        return loss.item()

    # ------------------------------------------------------------------
    # BaseTrainer interface
    # ------------------------------------------------------------------

    def update_model(self, batch_data: list) -> float:
        """Consume a batch of episodes and return scalar loss."""
        if self.loss_type == 'L0':
            return self._compute_l0_loss(batch_data)
        else:
            # For L1/L2, episodes were already stored in buffer by the
            # overridden train() loop. Compute contrastive loss from buffer.
            return self._compute_contrastive_loss()

    def update_params(self, params: Dict):
        if 'lr' in params:
            for pg in self.optimizer.param_groups:
                pg['lr'] = params['lr']
            print(f"Learning rate updated to: {params['lr']}")
        if 'beta' in params:
            self.beta = params['beta']
            self.beta_calibrated = True
        if 'temperature' in params:
            self.temperature = params['temperature']

    # ------------------------------------------------------------------
    # Custom training loop
    # ------------------------------------------------------------------

    def train(self, num_episodes: int, batch_size: int = 8,
              save_path: str = None, callback=None, start_episode: int = 0,
              episodes_per_step: int = 8):
        """Training loop with replay-buffer-based contrastive updates.

        For L0: uses the default BaseTrainer loop (TD on chains).
        For L1/L2: collects episodes, adds to buffer, samples contrastive
        batch from buffer each step.
        """
        if self.loss_type == 'L0':
            # L0 uses standard TD(0) training on episode chains
            super().train(num_episodes=num_episodes, batch_size=batch_size,
                          save_path=save_path, callback=callback,
                          start_episode=start_episode)
            return

        # L1/L2: contrastive training with replay buffer
        self.agent.set_train_mode(True)
        self.stop_requested = False

        num_steps = num_episodes // episodes_per_step
        for step in range(num_steps):
            if self.stop_requested:
                print("Training stop requested.")
                break

            episode = start_episode + (step + 1) * episodes_per_step

            # Collect fresh data
            self._collect_and_store(num_episodes=episodes_per_step)

            # Contrastive update from buffer
            loss = self._compute_contrastive_loss()
            self.loss_history.append(loss)

            if callback:
                callback({
                    "episode": episode,
                    "loss": loss,
                    "buffer_size": len(self.replay_buffer),
                    "type": "batch_update",
                })

            if step < 3 or (step + 1) % (num_steps // 20 + 1) == 0:
                print(f"Step {step+1}/{num_steps} (ep {episode}), "
                      f"Loss: {loss:.6f}, Buffer: {len(self.replay_buffer)}")

            # Periodic evaluation
            if episode % self.eval_interval == 0 and self.loss_type == 'L0':
                avg_chips = self.evaluate(num_games=self.eval_num_games)
                print(f"Episode {episode}, Avg Chips/Round: {avg_chips:+.2f}")

        if save_path:
            import os
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            self.agent.save_model(save_path)
            print(f"Model saved to {save_path}")

    def get_loss_history(self) -> List[float]:
        return self.loss_history
