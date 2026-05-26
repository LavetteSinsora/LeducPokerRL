"""Trainer for dual-axis subspace-partitioned representation learning (v5).

Collects data via self-play with a frozen ValueBasedAgent, stores
(state_encoding, terminal_reward, opponent_hand_label) tuples in a replay
buffer, and trains the encoder using SUBSPACE-PARTITIONED losses:

    z_reward = embeddings[:, 0:4]   # reward subspace
    z_hand   = embeddings[:, 4:8]   # hand subspace

    L_reward = SoftDistanceL1Loss()(z_reward, rewards)
    L_hand   = SupConLoss()(z_hand, hand_labels)
    L_var    = VICRegVarianceLoss()(embeddings)  # full 8-dim

    L_total  = L_reward + lambda_hand * L_hand + lambda_var * L_var

Key difference from v3/v4: Each loss is applied ONLY to its own 4-dim slice.
Gradients from the reward loss flow only through dims 0-3 of the output layer.
Gradients from the hand loss flow only through dims 4-7. No interference is
possible by construction — this is structural isolation, not loss normalization.
"""

import os
import random
from collections import deque
from typing import Dict, List, Tuple

import torch
import torch.optim as optim

from preliminary_experiments.promoted_registry.base_trainer import BaseTrainer
from agents.value_based.agent import ValueBasedAgent
from engine.leduc_game import LeducGame

from experiments.representation_learning.dual_axis_repr_v5.agent import DualAxisV5Agent
from experiments.representation_learning.dual_axis_repr_v5.losses import (
    SoftDistanceL1Loss,
    SupConLoss,
    VICRegVarianceLoss,
)


HAND_LABEL_MAP = {'J': 0, 'Q': 1, 'K': 2}


class DualAxisV5ReplayBuffer:
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


class DualAxisV5Trainer(BaseTrainer):
    """Trains DualAxisV5Agent using subspace-partitioned hybrid loss.

    The key innovation: each loss operates on only its own 4-dim slice:
      - SoftDistanceL1Loss applied to z[:, 0:4] (reward subspace only)
      - SupConLoss applied to z[:, 4:8] (hand subspace only)
      - VICRegVarianceLoss applied to full z[:, 0:8] (collapse prevention)

    This provides structural gradient isolation — no normalization heuristics
    needed. The reward and hand losses literally cannot interfere because they
    backpropagate through disjoint output dimensions.

    Args:
        agent: the encoder agent
        data_agent_path: path to trained value-based checkpoint for data collection
        learning_rate: Adam LR
        batch_size: samples drawn from replay buffer per update
        buffer_capacity: replay buffer size
        temperature: SupCon temperature tau for hand loss (default 0.07)
        lambda_hand: weight for hand SupCon loss (default 1.0)
        lambda_var: weight for VICReg variance loss (default 0.1)
        episodes_per_step: episodes collected per training step
    """

    def __init__(self, agent: DualAxisV5Agent,
                 data_agent_path: str = 'agents/value_based/checkpoint.pt',
                 learning_rate: float = 1e-4,
                 batch_size: int = 256,
                 buffer_capacity: int = 5000,
                 temperature: float = 0.07,
                 lambda_hand: float = 1.0,
                 lambda_var: float = 0.1,
                 episodes_per_step: int = 8):

        super().__init__(agent, eval_interval=500, eval_num_games=100)

        self.batch_size = batch_size
        self.temperature = temperature
        self.lambda_hand = lambda_hand
        self.lambda_var = lambda_var
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
        self.replay_buffer = DualAxisV5ReplayBuffer(capacity=buffer_capacity)
        self.episode_counter = 0

        # Loss functions
        self.l1_reward_loss_fn = SoftDistanceL1Loss()
        self.supcon_hand_loss_fn = SupConLoss(temperature=temperature)
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
        state_buffer: List[Tuple[torch.Tensor, int, int]] = []  # (encoded, player_id, opp_label)

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
    # Loss computation — SUBSPACE PARTITIONED
    # ------------------------------------------------------------------

    def _compute_loss(self) -> Tuple[float, float, float, float]:
        """Compute subspace-partitioned hybrid loss from replay buffer.

        Critical structural change vs v3/v4:
          - z_reward = embeddings[:, 0:4]  — reward subspace
          - z_hand   = embeddings[:, 4:8]  — hand subspace
          Each loss backpropagates through ONLY its own 4 output dimensions.
          Gradient interference is impossible by construction.

        Returns:
            (total_loss, reward_loss, hand_loss, var_loss) as floats
        """
        if len(self.replay_buffer) < self.batch_size:
            return 0.0, 0.0, 0.0, 0.0

        states, rewards, hand_labels = self.replay_buffer.sample(self.batch_size)

        # Forward through encoder — full 8-dim output
        embeddings = self.agent.encoder(states)  # (N, 8)

        # Split into disjoint subspaces
        z_reward = embeddings[:, 0:4]   # reward subspace — only these dims get reward gradients
        z_hand   = embeddings[:, 4:8]   # hand subspace   — only these dims get hand gradients

        # Apply each loss ONLY to its own subspace
        l_reward = self.l1_reward_loss_fn(z_reward, rewards)
        l_hand   = self.supcon_hand_loss_fn(z_hand, hand_labels)
        # VICReg applied to full embedding to prevent collapse in either subspace
        l_var    = self.vicreg_loss_fn(embeddings)

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
            self.supcon_hand_loss_fn.temperature = params['temperature']

    # ------------------------------------------------------------------
    # Custom training loop
    # ------------------------------------------------------------------

    def train(self, num_episodes: int, save_path: str = None, callback=None,
              start_episode: int = 0):
        """Training loop with replay-buffer-based subspace-partitioned loss updates.

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

            # Subspace-partitioned loss update from buffer
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
    # Diagnostics — extended for subspace analysis
    # ------------------------------------------------------------------

    def run_diagnostics(self) -> dict:
        """Run post-training diagnostics with subspace-specific metrics.

        Returns dict with keys:
          Full embedding:
            - effective_dim_80: PCA components explaining 80% variance (full 8-dim)
            - effective_dim_90: PCA components explaining 90% variance
            - pca_explained_variance: per-component ratios (full)
            - opp_hand_accuracy_full: linear probe hand accuracy from full embedding
            - reward_spearman_rho_full: Spearman rho, full embedding dist vs |delta_reward|
            - reward_bin_accuracy: 5-fold CV logistic regression on reward bins

          Reward subspace (dims 0:4):
            - reward_subspace_effective_dim_80: PCA effective dim (80%)
            - reward_subspace_spearman_rho: Spearman rho from reward subspace only
            - reward_subspace_hand_accuracy: linear probe on hand labels (cross-contamination check)

          Hand subspace (dims 4:8):
            - hand_subspace_effective_dim_80: PCA effective dim (80%)
            - hand_subspace_opp_hand_accuracy: linear probe hand accuracy
            - hand_subspace_spearman_rho: Spearman rho from hand subspace (cross-contamination check)
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
            full_embeddings = self.agent.encoder(states).numpy()   # (N, 8)
        self.agent.set_train_mode(True)

        emb_reward = full_embeddings[:, 0:4]   # reward subspace
        emb_hand   = full_embeddings[:, 4:8]   # hand subspace

        hand_labels_np = hand_labels.numpy()
        rewards_np = rewards.numpy()

        def compute_pca_effective_dim(emb, threshold=0.80):
            pca = PCA()
            pca.fit(emb)
            cumvar = np.cumsum(pca.explained_variance_ratio_)
            return int(np.searchsorted(cumvar, threshold)) + 1, pca.explained_variance_ratio_

        def compute_spearman(emb, rewards_np, subsample=2000):
            N = len(emb)
            if N * (N - 1) // 2 > subsample:
                rng = np.random.default_rng(42)
                idx_i = rng.integers(0, N, size=subsample)
                idx_j = rng.integers(0, N, size=subsample)
                same = idx_i == idx_j
                idx_j[same] = (idx_j[same] + 1) % N
            else:
                from itertools import combinations
                pairs = list(combinations(range(N), 2))
                if not pairs:
                    return 0.0, 1.0
                idx_i = np.array([p[0] for p in pairs])
                idx_j = np.array([p[1] for p in pairs])
            embed_dists = np.linalg.norm(emb[idx_i] - emb[idx_j], axis=1)
            reward_dists = np.abs(rewards_np[idx_i] - rewards_np[idx_j])
            rho, p = spearmanr(embed_dists, reward_dists)
            return float(rho), float(p)

        def linear_probe_accuracy(emb, labels, cv=5):
            clf = LogisticRegression(max_iter=1000, random_state=42)
            scores = cross_val_score(clf, emb, labels, cv=cv, scoring='accuracy')
            return float(scores.mean())

        # --- Full embedding ---
        eff_dim_80, explained_var = compute_pca_effective_dim(full_embeddings, 0.80)
        eff_dim_90, _ = compute_pca_effective_dim(full_embeddings, 0.90)
        rho_full, p_full = compute_spearman(full_embeddings, rewards_np)
        hand_acc_full = linear_probe_accuracy(full_embeddings, hand_labels_np)

        # Reward bins for bin-accuracy probe
        thresholds = [-2.0, -0.5, 0.5, 2.0]
        reward_bins_np = np.zeros(len(rewards_np), dtype=int)
        for t in thresholds:
            reward_bins_np += (rewards_np >= t).astype(int)
        reward_bin_acc = linear_probe_accuracy(full_embeddings, reward_bins_np)

        # --- Reward subspace (dims 0:4) ---
        eff_dim_reward, _ = compute_pca_effective_dim(emb_reward, 0.80)
        rho_reward, p_reward = compute_spearman(emb_reward, rewards_np)
        hand_acc_reward_subspace = linear_probe_accuracy(emb_reward, hand_labels_np)

        # --- Hand subspace (dims 4:8) ---
        eff_dim_hand, _ = compute_pca_effective_dim(emb_hand, 0.80)
        rho_hand_subspace, _ = compute_spearman(emb_hand, rewards_np)
        hand_acc_hand_subspace = linear_probe_accuracy(emb_hand, hand_labels_np)

        results = {
            # Full embedding
            'effective_dim_80': eff_dim_80,
            'effective_dim_90': eff_dim_90,
            'pca_explained_variance': explained_var.tolist(),
            'opp_hand_accuracy': hand_acc_full,          # legacy key for compatibility
            'opp_hand_accuracy_full': hand_acc_full,
            'reward_spearman_rho': rho_full,             # legacy key
            'reward_spearman_rho_full': rho_full,
            'reward_spearman_p_full': p_full,
            'reward_bin_accuracy': reward_bin_acc,
            # Reward subspace (dims 0:4)
            'reward_subspace_effective_dim_80': eff_dim_reward,
            'reward_subspace_spearman_rho': rho_reward,
            'reward_subspace_spearman_p': p_reward,
            'reward_subspace_hand_accuracy': hand_acc_reward_subspace,  # cross-contamination
            # Hand subspace (dims 4:8)
            'hand_subspace_effective_dim_80': eff_dim_hand,
            'hand_subspace_opp_hand_accuracy': hand_acc_hand_subspace,
            'hand_subspace_spearman_rho': rho_hand_subspace,             # cross-contamination
            # Meta
            'buffer_size': len(self.replay_buffer),
        }

        return results
