"""Contrastive representation encoder for Leduc Hold'em states.

This module defines the encoder network that maps 15-dim state observations
to 8-dim embeddings. It reuses the observation encoding logic from
ValueBasedAgent but replaces the value head with a contrastive embedding.
"""

import torch
import torch.nn as nn

from agents.base import BaseAgent
from agents.value_based.agent import ValueBasedAgent
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


class ValueHead(nn.Module):
    """Linear value head for L0 control (TD(0) baseline)."""

    def __init__(self, embedding_dim: int = 8):
        super().__init__()
        self.linear = nn.Linear(embedding_dim, 1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.linear(z)


class ContrastiveReprAgent(BaseAgent):
    """Agent that produces contrastive state embeddings.

    In Phase 1, this agent is used only for representation learning
    (no action selection). The encoder maps observations to embeddings
    that are trained via contrastive losses.

    For L0 control, a linear value head is attached for TD(0) training,
    and action selection uses 1-step lookahead (same as ValueBasedAgent).
    """

    CARD_MAP = {'J': 0, 'Q': 1, 'K': 2}
    MAX_CHIPS = 13

    def __init__(self, input_size: int = 15, embedding_dim: int = 8,
                 model_path: str = None, use_value_head: bool = False):
        self.input_size = input_size
        self.embedding_dim = embedding_dim
        self.use_value_head = use_value_head
        self.train_mode = False

        self.encoder = ContrastiveEncoder(input_size, embedding_dim=embedding_dim)
        self.value_head = ValueHead(embedding_dim) if use_value_head else None

        if model_path:
            self.load_model(model_path)

    def set_train_mode(self, mode: bool):
        self.train_mode = mode
        self.encoder.train(mode)
        if self.value_head:
            self.value_head.train(mode)

    def save_model(self, path: str) -> None:
        state = {'encoder': self.encoder.state_dict()}
        if self.value_head:
            state['value_head'] = self.value_head.state_dict()
        torch.save(state, path)

    def load_model(self, path: str) -> None:
        state = torch.load(path)
        self.encoder.load_state_dict(state['encoder'])
        if self.value_head and 'value_head' in state:
            self.value_head.load_state_dict(state['value_head'])

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

    def _get_value(self, obs: Observation, viewer_id: int) -> float:
        """Get value estimate (only works with value head, for L0)."""
        if not self.value_head:
            return 0.0
        encoded = self.encode_observation(obs, viewer_id=viewer_id)
        with torch.no_grad():
            z = self.encoder(encoded)
            return self.value_head(z).item()

    def select_action(self, obs: Observation) -> Action:
        """Select action via 1-step lookahead (only meaningful for L0 control)."""
        if not self.use_value_head:
            legal = obs.legal_actions
            return legal[torch.randint(len(legal), (1,)).item()] if legal else Action.FOLD

        from engine.leduc_game import LeducGame
        current_p = obs.current_player
        best_val = float('-inf')
        best_action = Action.FOLD

        for action in obs.legal_actions:
            post_obs, done = LeducGame.simulate_action(obs, action)
            if done and action == Action.FOLD:
                val = -float(obs.pot[current_p])
            else:
                val = self._get_value(post_obs, viewer_id=current_p)
            if val > best_val:
                best_val = val
                best_action = action

        return best_action
