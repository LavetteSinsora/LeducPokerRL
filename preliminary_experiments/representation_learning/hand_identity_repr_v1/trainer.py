"""Trainer for hand-identity representation learning.

Collects data via self-play with a frozen ValueBasedAgent,
stores (state_encoding, opponent_hand_label) tuples in a replay buffer,
and trains the encoder using triplet loss and/or cross-entropy loss.

The key difference from contrastive_repr_v1: supervision signal is the
opponent's actual hand identity (J=0, Q=1, K=2) rather than terminal reward.
"""

import os
import random
from collections import deque
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim

from preliminary_experiments.promoted_registry.base_trainer import BaseTrainer
from agents.value_based.agent import ValueBasedAgent
from engine.leduc_game import LeducGame, Action

from experiments.representation_learning.hand_identity_repr_v1.agent import HandIdentityReprAgent
from experiments.representation_learning.hand_identity_repr_v1.losses import TripletLoss, CrossEntropyHandLoss


HAND_LABEL_MAP = {'J': 0, 'Q': 1, 'K': 2}


class HandLabelReplayBuffer:
    """Stores (state_encoding, opponent_hand_label) tuples."""

    def __init__(self, capacity: int = 5000):
        self.capacity = capacity
        self.buffer: deque = deque(maxlen=capacity)

    def add(self, state: torch.Tensor, hand_label: int):
        self.buffer.append((state.detach(), hand_label))

    def sample(self, n: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample n items from the buffer.

        Returns:
            (states [N, 15], hand_labels [N])
        """
        n = min(n, len(self.buffer))
        items = random.sample(list(self.buffer), n)
        states = torch.cat([s for s, _ in items], dim=0)       # (N, 15)
        labels = torch.tensor([l for _, l in items], dtype=torch.long)  # (N,)
        return states, labels

    def get_all(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return all buffer contents as tensors."""
        if not self.buffer:
            return torch.zeros(0, 15), torch.zeros(0, dtype=torch.long)
        states = torch.cat([s for s, _ in self.buffer], dim=0)
        labels = torch.tensor([l for _, l in self.buffer], dtype=torch.long)
        return states, labels

    def __len__(self):
        return len(self.buffer)


class HandIdentityReprTrainer(BaseTrainer):
    """Trains HandIdentityReprAgent using hand-label-supervised losses.

    Args:
        agent: the encoder agent
        data_agent_path: path to trained value-based checkpoint for data collection
        loss_type: 'triplet', 'ce', or 'both'
        learning_rate: Adam LR
        batch_size: samples drawn from replay buffer per update
        buffer_capacity: replay buffer size
        margin: triplet loss margin
        episodes_per_step: episodes collected per training step
    """

    def __init__(self, agent: HandIdentityReprAgent,
                 data_agent_path: str = 'agents/value_based/checkpoint.pt',
                 loss_type: str = 'triplet',
                 learning_rate: float = 1e-4,
                 batch_size: int = 256,
                 buffer_capacity: int = 5000,
                 margin: float = 1.0,
                 episodes_per_step: int = 8):

        super().__init__(agent, eval_interval=500, eval_num_games=100)

        self.loss_type = loss_type.lower()
        assert self.loss_type in ('triplet', 'ce', 'both'), \
            f"Unknown loss type: {loss_type}"

        self.batch_size = batch_size
        self.margin = margin
        self.episodes_per_step = episodes_per_step

        # Data collection agent (frozen value-based policy)
        if os.path.exists(data_agent_path):
            self.data_agent = ValueBasedAgent(model_path=data_agent_path)
            print(f"Loaded data agent from {data_agent_path}")
        else:
            # Try fallback path
            fallback = 'agents/value_based/checkpoint.pt'
            if os.path.exists(fallback):
                self.data_agent = ValueBasedAgent(model_path=fallback)
                print(f"Loaded data agent from fallback {fallback}")
            else:
                self.data_agent = ValueBasedAgent()
                print("No checkpoint found — using fresh ValueBasedAgent for data collection")
        self.data_agent.set_train_mode(False)

        self.game = LeducGame()
        self.replay_buffer = HandLabelReplayBuffer(capacity=buffer_capacity)
        self.episode_counter = 0

        # Loss functions
        self.triplet_loss_fn = TripletLoss(margin=margin)
        self.ce_loss_fn = CrossEntropyHandLoss(
            embedding_dim=agent.embedding_dim
        ) if self.loss_type in ('ce', 'both') else None

        # Optimizer
        params = list(agent.encoder.parameters())
        if self.ce_loss_fn is not None:
            params += list(self.ce_loss_fn.head.parameters())
        self.optimizer = optim.Adam(params, lr=learning_rate)

        # Logging
        self.loss_history: List[float] = []

    # ------------------------------------------------------------------
    # Data collection
    # ------------------------------------------------------------------

    def collect_episode(self) -> List[Tuple[torch.Tensor, int]]:
        """Play one self-play episode and return (state_encoding, opponent_hand_label) pairs.

        For each state visited by player P, the opponent is (1-P), and the
        opponent's hand label is recorded as the supervision signal.
        """
        self.game.reset()
        samples = []

        while not self.game.is_finished:
            current_player = self.game.current_player
            opponent = 1 - current_player

            # Opponent's hand is the supervision signal
            opponent_hand_str = self.game.player_hands[opponent]
            opponent_label = HAND_LABEL_MAP.get(opponent_hand_str)
            if opponent_label is None:
                # Shouldn't happen in normal play, skip
                obs = self.game.get_observation(viewer_id=current_player)
                action = self.data_agent.select_action(obs)
                self.game.step(action)
                continue

            obs = self.game.get_observation(viewer_id=current_player)
            encoded = self.agent.encode_observation(obs, viewer_id=current_player)
            samples.append((encoded, opponent_label))

            action = self.data_agent.select_action(obs)
            self.game.step(action)

        return samples

    def _collect_and_store(self, num_episodes: int = 8):
        """Collect episodes and add to replay buffer."""
        for _ in range(num_episodes):
            samples = self.collect_episode()
            self.episode_counter += 1
            for state_enc, label in samples:
                self.replay_buffer.add(state_enc, label)

    # ------------------------------------------------------------------
    # Loss computation
    # ------------------------------------------------------------------

    def _compute_loss(self) -> float:
        """Compute loss from replay buffer samples."""
        if len(self.replay_buffer) < self.batch_size:
            return 0.0

        states, labels = self.replay_buffer.sample(self.batch_size)

        # Forward through encoder
        z = self.agent.encoder(states)  # (N, D)

        total_loss = torch.tensor(0.0)

        if self.loss_type in ('triplet', 'both'):
            total_loss = total_loss + self.triplet_loss_fn(z, labels)

        if self.loss_type in ('ce', 'both'):
            total_loss = total_loss + self.ce_loss_fn(z, labels)

        if total_loss.requires_grad:
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()

        return total_loss.item()

    # ------------------------------------------------------------------
    # BaseTrainer interface
    # ------------------------------------------------------------------

    def update_model(self, batch_data: list) -> float:
        """Consume batch data and return scalar loss (for BaseTrainer compat)."""
        return self._compute_loss()

    def update_params(self, params: Dict):
        if 'lr' in params:
            for pg in self.optimizer.param_groups:
                pg['lr'] = params['lr']
            print(f"Learning rate updated to: {params['lr']}")
        if 'margin' in params:
            self.triplet_loss_fn.margin = params['margin']

    # ------------------------------------------------------------------
    # Custom training loop
    # ------------------------------------------------------------------

    def train(self, num_episodes: int, save_path: str = None, callback=None,
              start_episode: int = 0):
        """Training loop with replay-buffer-based updates.

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

            # Loss update from buffer
            loss = self._compute_loss()
            self.loss_history.append(loss)

            if callback:
                callback({
                    "episode": episode,
                    "loss": loss,
                    "buffer_size": len(self.replay_buffer),
                    "type": "batch_update",
                })

            if step < 3 or (step + 1) % max(1, num_steps // 20) == 0:
                print(f"Step {step+1}/{num_steps} (ep {episode}), "
                      f"Loss: {loss:.6f}, Buffer: {len(self.replay_buffer)}")

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            self.agent.save_model(save_path)
            print(f"Model saved to {save_path}")

    def get_loss_history(self) -> List[float]:
        return self.loss_history

    def run_linear_probe(self) -> float:
        """Train logistic regression on (embedding -> opponent_hand_label).

        Returns:
            Cross-validated accuracy score
        """
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import cross_val_score

        if len(self.replay_buffer) < 50:
            print("Buffer too small for linear probe.")
            return 0.0

        states, labels = self.replay_buffer.get_all()

        with torch.no_grad():
            self.agent.set_train_mode(False)
            embeddings = self.agent.encoder(states).numpy()
            self.agent.set_train_mode(True)

        labels_np = labels.numpy()

        clf = LogisticRegression(max_iter=1000, random_state=42)
        scores = cross_val_score(clf, embeddings, labels_np, cv=5, scoring='accuracy')
        return float(scores.mean())
