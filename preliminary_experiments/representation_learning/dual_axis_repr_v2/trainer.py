"""Trainer for dual-axis SupCon representation learning.

Collects data via self-play with a frozen ValueBasedAgent, stores
(state_encoding, terminal_reward, opponent_hand_label) tuples in a replay
buffer, and trains the encoder using two independent SupCon losses plus
VICReg variance regularization.

Loss:
    L_total = L_SupCon(reward_bins) + lambda_hand * L_SupCon(hand_labels)
              + lambda_var * L_VICReg
"""

import os
import random
from collections import deque
from typing import Dict, List, Optional, Tuple

import torch
import torch.optim as optim

from preliminary_experiments.promoted_registry.base_trainer import BaseTrainer
from agents.value_based.agent import ValueBasedAgent
from engine.leduc_game import LeducGame

from experiments.representation_learning.dual_axis_repr_v2.agent import DualAxisV2Agent
from experiments.representation_learning.dual_axis_repr_v2.losses import (
    SupConLoss,
    VICRegVarianceLoss,
    get_reward_bin_labels,
)


HAND_LABEL_MAP = {'J': 0, 'Q': 1, 'K': 2}


class DualAxisReplayBuffer:
    """Stores (state_encoding, terminal_reward, opponent_hand_label) tuples."""

    def __init__(self, capacity: int = 5000):
        self.capacity = capacity
        self.buffer: deque = deque(maxlen=capacity)

    def add(self, state: torch.Tensor, reward: float, hand_label: int):
        self.buffer.append((state.detach(), reward, hand_label))

    def sample(self, n: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample n items from the buffer.

        Returns:
            (states [N, 15], rewards [N] float, hand_labels [N] long)
        """
        n = min(n, len(self.buffer))
        items = random.sample(list(self.buffer), n)
        states = torch.cat([s for s, _, _ in items], dim=0)          # (N, 15)
        rewards = torch.tensor([r for _, r, _ in items], dtype=torch.float)  # (N,)
        hand_labels = torch.tensor([h for _, _, h in items], dtype=torch.long)   # (N,)
        return states, rewards, hand_labels

    def get_all(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return all buffer contents as tensors."""
        if not self.buffer:
            return (torch.zeros(0, 15), torch.zeros(0),
                    torch.zeros(0, dtype=torch.long))
        states = torch.cat([s for s, _, _ in self.buffer], dim=0)
        rewards = torch.tensor([r for _, r, _ in self.buffer], dtype=torch.float)
        hand_labels = torch.tensor([h for _, _, h in self.buffer], dtype=torch.long)
        return states, rewards, hand_labels

    def __len__(self):
        return len(self.buffer)


class DualAxisV2Trainer(BaseTrainer):
    """Trains DualAxisV2Agent using two independent SupCon losses.

    Args:
        agent: the encoder agent
        data_agent_path: path to trained value-based checkpoint for data collection
        learning_rate: Adam LR
        batch_size: samples drawn from replay buffer per update
        buffer_capacity: replay buffer size
        temperature: SupCon temperature tau (default 0.07)
        lambda_hand: weight for hand SupCon loss (default 1.0)
        lambda_var: weight for VICReg variance loss (default 0.1)
        n_bins: number of reward bins (default 5)
        episodes_per_step: episodes collected per training step
    """

    def __init__(self, agent: DualAxisV2Agent,
                 data_agent_path: str = 'agents/value_based/checkpoint.pt',
                 learning_rate: float = 1e-4,
                 batch_size: int = 256,
                 buffer_capacity: int = 5000,
                 temperature: float = 0.07,
                 lambda_hand: float = 1.0,
                 lambda_var: float = 0.1,
                 n_bins: int = 5,
                 episodes_per_step: int = 8):

        super().__init__(agent, eval_interval=500, eval_num_games=100)

        self.batch_size = batch_size
        self.temperature = temperature
        self.lambda_hand = lambda_hand
        self.lambda_var = lambda_var
        self.n_bins = n_bins
        self.episodes_per_step = episodes_per_step

        # Data collection agent (frozen value-based policy)
        if os.path.exists(data_agent_path):
            self.data_agent = ValueBasedAgent(model_path=data_agent_path)
            print(f"Loaded data agent from {data_agent_path}")
        else:
            fallback = 'agents/value_based/checkpoint.pt'
            if os.path.exists(fallback):
                self.data_agent = ValueBasedAgent(model_path=fallback)
                print(f"Loaded data agent from fallback {fallback}")
            else:
                self.data_agent = ValueBasedAgent()
                print("No checkpoint found — using fresh ValueBasedAgent for data collection")
        self.data_agent.set_train_mode(False)

        self.game = LeducGame()
        self.replay_buffer = DualAxisReplayBuffer(capacity=buffer_capacity)
        self.episode_counter = 0

        # Loss functions
        self.supcon_loss_fn = SupConLoss(temperature=temperature)
        self.vicreg_loss_fn = VICRegVarianceLoss()

        # Optimizer
        self.optimizer = optim.Adam(agent.encoder.parameters(), lr=learning_rate)

        # Logging
        self.loss_history: List[float] = []
        self.loss_reward_history: List[float] = []
        self.loss_hand_history: List[float] = []
        self.loss_var_history: List[float] = []

    # ------------------------------------------------------------------
    # Data collection
    # ------------------------------------------------------------------

    def collect_episode(self) -> List[Tuple[torch.Tensor, float, int]]:
        """Play one self-play episode and return (state_encoding, reward, opp_hand_label) tuples.

        For each game state visited by player P, we record:
          - state_encoding: 15-dim encoded observation from P's perspective
          - terminal_reward: chip outcome for P at game end
          - opponent_hand_label: opponent's card as int (J=0, Q=1, K=2)
        """
        self.game.reset()
        samples: List[Tuple[torch.Tensor, float, int]] = []

        # Collect state observations during game play
        state_buffer: List[Tuple[torch.Tensor, int]] = []  # (encoded, player_id)

        while not self.game.is_finished:
            current_player = self.game.current_player
            opponent = 1 - current_player

            # Opponent's hand is the supervision signal
            opponent_hand_str = self.game.player_hands[opponent]
            opponent_label = HAND_LABEL_MAP.get(opponent_hand_str)

            obs = self.game.get_observation(viewer_id=current_player)
            encoded = self.agent.encode_observation(obs, viewer_id=current_player)

            if opponent_label is not None:
                state_buffer.append((encoded, current_player, opponent_label))

            action = self.data_agent.select_action(obs)
            self.game.step(action)

        rewards = self.game.get_reward()

        # Assign terminal reward to each recorded state
        for encoded, player_id, opp_label in state_buffer:
            terminal_reward = float(rewards[player_id])
            samples.append((encoded, terminal_reward, opp_label))

        return samples

    def _collect_and_store(self, num_episodes: int = 8):
        """Collect episodes and add all states to replay buffer."""
        for _ in range(num_episodes):
            samples = self.collect_episode()
            self.episode_counter += 1
            for state_enc, reward, hand_label in samples:
                self.replay_buffer.add(state_enc, reward, hand_label)

    # ------------------------------------------------------------------
    # Loss computation
    # ------------------------------------------------------------------

    def _compute_loss(self) -> Tuple[float, float, float, float]:
        """Compute dual-axis SupCon loss from replay buffer samples.

        Returns:
            (total_loss, reward_loss, hand_loss, var_loss) as floats
        """
        if len(self.replay_buffer) < self.batch_size:
            return 0.0, 0.0, 0.0, 0.0

        states, rewards, hand_labels = self.replay_buffer.sample(self.batch_size)

        # Forward through encoder
        embeddings = self.agent.encoder(states)  # (N, D)

        # Reward bins
        reward_bin_labels = get_reward_bin_labels(rewards, n_bins=self.n_bins)

        # Compute individual losses
        l_reward = self.supcon_loss_fn(embeddings, reward_bin_labels)
        l_hand = self.supcon_loss_fn(embeddings, hand_labels)
        l_var = self.vicreg_loss_fn(embeddings)

        total_loss = l_reward + self.lambda_hand * l_hand + self.lambda_var * l_var

        if total_loss.requires_grad:
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()

        return (total_loss.item(), l_reward.item(),
                l_hand.item(), l_var.item())

    # ------------------------------------------------------------------
    # BaseTrainer interface
    # ------------------------------------------------------------------

    def update_model(self, batch_data: list) -> float:
        """Consume batch data and return scalar loss (for BaseTrainer compat)."""
        total, _, _, _ = self._compute_loss()
        return total

    def update_params(self, params: Dict):
        if 'lr' in params:
            for pg in self.optimizer.param_groups:
                pg['lr'] = params['lr']
            print(f"Learning rate updated to: {params['lr']}")
        if 'lambda_hand' in params:
            self.lambda_hand = params['lambda_hand']
        if 'lambda_var' in params:
            self.lambda_var = params['lambda_var']
        if 'temperature' in params:
            self.temperature = params['temperature']
            self.supcon_loss_fn.temperature = params['temperature']

    # ------------------------------------------------------------------
    # Custom training loop
    # ------------------------------------------------------------------

    def train(self, num_episodes: int, save_path: str = None, callback=None,
              start_episode: int = 0):
        """Training loop with replay-buffer-based dual-axis SupCon updates.

        Args:
            num_episodes: total episodes to collect
            save_path: where to save the model checkpoint
            callback: progress callback function
            start_episode: episode counter offset
        """
        self.agent.set_train_mode(True)
        self.stop_requested = False

        num_steps = num_episodes // self.episodes_per_step
        for step in range(num_steps):
            if self.stop_requested:
                print("Training stop requested.")
                break

            episode = start_episode + (step + 1) * self.episodes_per_step

            # Collect fresh data
            self._collect_and_store(num_episodes=self.episodes_per_step)

            # Dual-axis SupCon update from buffer
            total_loss, r_loss, h_loss, v_loss = self._compute_loss()
            self.loss_history.append(total_loss)
            self.loss_reward_history.append(r_loss)
            self.loss_hand_history.append(h_loss)
            self.loss_var_history.append(v_loss)

            if callback:
                callback({
                    "episode": episode,
                    "loss": total_loss,
                    "loss_reward": r_loss,
                    "loss_hand": h_loss,
                    "loss_var": v_loss,
                    "buffer_size": len(self.replay_buffer),
                    "type": "batch_update",
                })

            if step < 3 or (step + 1) % max(1, num_steps // 20) == 0:
                print(f"Step {step+1}/{num_steps} (ep {episode}), "
                      f"Loss: {total_loss:.4f} "
                      f"(R:{r_loss:.4f} H:{h_loss:.4f} V:{v_loss:.4f}), "
                      f"Buffer: {len(self.replay_buffer)}")

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            self.agent.save_model(save_path)
            print(f"Model saved to {save_path}")

    def get_loss_history(self) -> List[float]:
        return self.loss_history

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def run_diagnostics(self) -> dict:
        """Run post-training diagnostics on all buffered embeddings.

        Returns:
            dict with keys:
              - effective_dim_80: number of PCA components explaining 80% variance
              - effective_dim_90: number of PCA components explaining 90% variance
              - pca_explained_variance: list of per-component explained variance ratios
              - opp_hand_accuracy: 5-fold CV logistic regression accuracy (opponent hand)
              - reward_bin_accuracy: 5-fold CV logistic regression accuracy (reward bin)
              - reward_spearman_rho: Spearman rho between pairwise embed dist and |delta_reward|
        """
        import numpy as np
        from sklearn.decomposition import PCA
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import cross_val_score
        from scipy.stats import spearmanr

        if len(self.replay_buffer) < 50:
            print("Buffer too small for diagnostics.")
            return {}

        states, rewards, hand_labels = self.replay_buffer.get_all()

        self.agent.set_train_mode(False)
        with torch.no_grad():
            embeddings = self.agent.encoder(states).numpy()
        self.agent.set_train_mode(True)

        hand_labels_np = hand_labels.numpy()
        rewards_np = rewards.numpy()

        # --- PCA effective dimensionality ---
        pca = PCA()
        pca.fit(embeddings)
        explained_var = pca.explained_variance_ratio_
        cumvar = np.cumsum(explained_var)
        effective_dim_80 = int(np.searchsorted(cumvar, 0.80)) + 1
        effective_dim_90 = int(np.searchsorted(cumvar, 0.90)) + 1

        # --- Linear probe: opponent hand ---
        clf_hand = LogisticRegression(max_iter=1000, random_state=42)
        hand_scores = cross_val_score(clf_hand, embeddings, hand_labels_np,
                                      cv=5, scoring='accuracy')
        opp_hand_accuracy = float(hand_scores.mean())

        # --- Linear probe: reward bins ---
        reward_bins_np = get_reward_bin_labels(
            torch.tensor(rewards_np)
        ).numpy()
        clf_reward = LogisticRegression(max_iter=1000, random_state=42)
        reward_scores = cross_val_score(clf_reward, embeddings, reward_bins_np,
                                        cv=5, scoring='accuracy')
        reward_bin_accuracy = float(reward_scores.mean())

        # --- Spearman rho: pairwise embedding dist vs |delta_reward| ---
        N = len(embeddings)
        subsample = 2000
        if N * (N - 1) // 2 > subsample:
            # Subsample pairs
            rng = np.random.default_rng(42)
            idx_i = rng.integers(0, N, size=subsample)
            idx_j = rng.integers(0, N, size=subsample)
            # Avoid self-pairs
            same = idx_i == idx_j
            idx_j[same] = (idx_j[same] + 1) % N

            embed_dists = np.linalg.norm(embeddings[idx_i] - embeddings[idx_j], axis=1)
            reward_dists = np.abs(rewards_np[idx_i] - rewards_np[idx_j])
        else:
            # All pairs
            from itertools import combinations
            pairs = list(combinations(range(N), 2))
            if not pairs:
                return {}
            idx_i = np.array([p[0] for p in pairs])
            idx_j = np.array([p[1] for p in pairs])
            embed_dists = np.linalg.norm(embeddings[idx_i] - embeddings[idx_j], axis=1)
            reward_dists = np.abs(rewards_np[idx_i] - rewards_np[idx_j])

        spearman_rho, spearman_p = spearmanr(embed_dists, reward_dists)

        results = {
            'effective_dim_80': effective_dim_80,
            'effective_dim_90': effective_dim_90,
            'pca_explained_variance': explained_var.tolist(),
            'opp_hand_accuracy': opp_hand_accuracy,
            'reward_bin_accuracy': reward_bin_accuracy,
            'reward_spearman_rho': float(spearman_rho),
            'reward_spearman_p': float(spearman_p),
            'buffer_size': len(self.replay_buffer),
        }

        return results
