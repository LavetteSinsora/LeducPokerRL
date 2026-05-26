"""Dual-axis contrastive representation encoder for Leduc Hold'em states.

Maps 15-dim state observations to 8-dim embeddings. Trained via a
joint reward+hand-identity contrastive loss that forces multi-axis
representation — encoding both strategic value geometry and opponent
hand identity simultaneously.

Architecture is identical to contrastive_repr_v1 and hand_identity_repr_v1.
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


class ClassificationHead(nn.Module):
    """Linear classification head for opponent hand prediction (8 -> 3)."""

    def __init__(self, embedding_dim: int = 8, num_classes: int = 3):
        super().__init__()
        self.linear = nn.Linear(embedding_dim, num_classes)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.linear(z)


class DualAxisReprAgent(BaseAgent):
    """Agent that produces dual-axis contrastive state embeddings.

    The encoder maps observations to embeddings trained via
    DualAxisContrastiveLoss + VICRegVarianceLoss, where positive pairs
    are defined by similarity on BOTH reward AND opponent hand identity.

    This is a representation learning agent — action selection is random
    during training data collection.

    Args:
        input_size: input feature dimension (default 15)
        embedding_dim: output embedding dimension (default 8)
        model_path: optional path to load checkpoint from
        use_classification_head: whether to include a linear probe head
    """

    CARD_MAP = {'J': 0, 'Q': 1, 'K': 2}
    MAX_CHIPS = 13

    def __init__(self, input_size: int = 15, embedding_dim: int = 8,
                 model_path: str = None, use_classification_head: bool = False):
        self.input_size = input_size
        self.embedding_dim = embedding_dim
        self.use_classification_head = use_classification_head

        self.encoder = ContrastiveEncoder(input_size, embedding_dim=embedding_dim)
        self.classification_head = (
            ClassificationHead(embedding_dim) if use_classification_head else None
        )

        if model_path:
            self.load_model(model_path)

    def set_train_mode(self, mode: bool):
        self.encoder.train(mode)
        if self.classification_head:
            self.classification_head.train(mode)

    def save_model(self, path: str) -> None:
        state = {'encoder': self.encoder.state_dict()}
        if self.classification_head:
            state['classification_head'] = self.classification_head.state_dict()
        torch.save(state, path)

    def load_model(self, path: str) -> None:
        state = torch.load(path, weights_only=True)
        self.encoder.load_state_dict(state['encoder'])
        if self.classification_head and 'classification_head' in state:
            self.classification_head.load_state_dict(state['classification_head'])

    def encode_observation(self, obs: Observation, viewer_id: int = None) -> torch.Tensor:
        """Encodes observation to 15-dim vector (same logic as ValueBasedAgent)."""
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
        """Get the embedding z for an observation."""
        encoded = self.encode_observation(obs, viewer_id=viewer_id)
        return self.encoder(encoded)

    def select_action(self, obs: Observation) -> Action:
        """Random action selection (this is a representation learning agent)."""
        legal = obs.legal_actions
        if not legal:
            return Action.FOLD
        idx = torch.randint(len(legal), (1,)).item()
        return legal[idx]
