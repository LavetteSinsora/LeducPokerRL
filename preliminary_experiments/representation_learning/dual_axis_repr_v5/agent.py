"""Dual-axis subspace-partitioned (v5) representation encoder for Leduc Hold'em states.

This module defines the encoder network that maps 15-dim state observations
to 8-dim embeddings trained via subspace-partitioned losses:
  - L1 soft-distance correlation loss applied ONLY to dims 0:4 (reward subspace)
  - SupCon loss over opponent hand identity applied ONLY to dims 4:8 (hand subspace)

The encoder architecture is identical to v3. The key innovation is in the
trainer: each loss touches only its own 4-dim slice of the output, so
gradients from the reward and hand objectives can never interfere.
"""

import torch
import torch.nn as nn

from agents.base import BaseAgent
from engine.observation import Observation
from engine.leduc_game import Action


class ContrastiveEncoder(nn.Module):
    """Maps 15-dim state encoding to 8-dim embedding."""

    def __init__(self, input_size: int = 15, hidden_size: int = 64,
                 embedding_dim: int = 8):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, embedding_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


class DualAxisV5Agent(BaseAgent):
    """Agent that produces dual-axis subspace-partitioned state embeddings.

    The 8-dim output is treated as two disjoint 4-dim subspaces:
      - dims 0:4 (z_reward): trained by L1 soft-distance reward loss only
      - dims 4:8 (z_hand):   trained by SupCon hand-identity loss only

    Action selection is random (this is a representation learning agent,
    not a policy agent). Self-play data is collected by a frozen
    ValueBasedAgent.

    Args:
        input_size: input feature dimension (default 15)
        embedding_dim: output embedding dimension (default 8)
        model_path: optional path to load checkpoint from
    """

    CARD_MAP = {'J': 0, 'Q': 1, 'K': 2}
    MAX_CHIPS = 13

    def __init__(self, input_size: int = 15, embedding_dim: int = 8,
                 model_path: str = None):
        self.input_size = input_size
        self.embedding_dim = embedding_dim

        self.encoder = ContrastiveEncoder(input_size, embedding_dim=embedding_dim)

        if model_path:
            self.load_model(model_path)

    def set_train_mode(self, mode: bool):
        self.encoder.train(mode)

    def save_model(self, path: str) -> None:
        torch.save({'encoder': self.encoder.state_dict()}, path)

    def load_model(self, path: str) -> None:
        state = torch.load(path, weights_only=True)
        self.encoder.load_state_dict(state['encoder'])

    def encode_observation(self, obs: Observation, viewer_id: int = None) -> torch.Tensor:
        """Encodes observation to 15-dim vector (same logic as ValueBasedAgent).

        Feature layout (15 dims):
          [0:3]   one-hot viewer hand card (J/Q/K)
          [3:7]   one-hot board card (J/Q/K/none)
          [7:9]   pot sizes (viewer, opponent) normalized by MAX_CHIPS
          [9]     is_to_act (1 if viewer is current player)
          [10]    viewer_id (0 or 1)
          [11]    current_round (0 or 1)
          [12]    is_finished (0 or 1)
          [13]    has_pair (1 if viewer hand matches board)
          [14]    raises_this_round / 2.0
        """
        if viewer_id is None:
            viewer_id = obs.current_player

        hand_idx = self.CARD_MAP.get(obs.player_hand)
        hand_vec = torch.zeros(3)
        if hand_idx is not None:
            hand_vec[hand_idx] = 1.0

        board_idx = self.CARD_MAP.get(obs.board, 3)
        board_vec = torch.zeros(4)
        board_vec[board_idx] = 1.0

        p0_pot, p1_pot = obs.pot
        pot_rel = [p0_pot, p1_pot] if viewer_id == 0 else [p1_pot, p0_pot]
        pot_vec = torch.tensor(pot_rel) / self.MAX_CHIPS

        features = torch.tensor([
            1.0 if viewer_id == obs.current_player else 0.0,
            float(viewer_id),
            float(obs.current_round),
            1.0 if obs.is_finished else 0.0,
            1.0 if (obs.board is not None and obs.player_hand == obs.board) else 0.0,
            obs.raises_this_round / 2.0,
        ])

        return torch.cat([hand_vec, board_vec, pot_vec, features]).unsqueeze(0)

    def get_embedding(self, obs: Observation, viewer_id: int = None) -> torch.Tensor:
        """Get the full 8-dim embedding z for an observation."""
        encoded = self.encode_observation(obs, viewer_id=viewer_id)
        return self.encoder(encoded)

    def get_reward_subspace(self, obs: Observation, viewer_id: int = None) -> torch.Tensor:
        """Get just the 4-dim reward subspace (dims 0:4)."""
        return self.get_embedding(obs, viewer_id=viewer_id)[:, 0:4]

    def get_hand_subspace(self, obs: Observation, viewer_id: int = None) -> torch.Tensor:
        """Get just the 4-dim hand subspace (dims 4:8)."""
        return self.get_embedding(obs, viewer_id=viewer_id)[:, 4:8]

    def select_action(self, obs: Observation) -> Action:
        """Random action selection (this is a representation learning agent)."""
        legal = obs.legal_actions
        if not legal:
            return Action.FOLD
        idx = torch.randint(len(legal), (1,)).item()
        return legal[idx]
