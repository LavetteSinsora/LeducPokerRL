"""
Actor-Critic Agent (A2C).

Shared trunk: 15 → 64 → 64
Policy head:  → 3 (action logits)
Value head:   → 1 (state value)

Trained with TD(0) advantage: A = r + gamma*V(s') - V(s)
"""

import torch
import torch.nn as nn

from agents.base import BaseAgent
from engine.leduc_game import Action
from engine.observation import Observation


class ActorCriticAgent(BaseAgent):
    """
    Advantage Actor-Critic (A2C).
    Shared trunk: 15 → 64 → 64
    Policy head: → 3 (action logits)
    Value head:  → 1 (state value)
    """

    CARD_MAP = {'J': 0, 'Q': 1, 'K': 2}
    MAX_CHIPS = 13

    def __init__(self, model_path: str = None):
        self.train_mode = False

        self.trunk = nn.Sequential(
            nn.Linear(15, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
        )
        self.policy_head = nn.Linear(64, 3)
        self.value_head = nn.Linear(64, 1)

        if model_path:
            self.load_model(model_path)
        self.trunk.eval()
        self.policy_head.eval()
        self.value_head.eval()

    def set_train_mode(self, mode: bool):
        self.train_mode = mode
        self.trunk.train(mode)
        self.policy_head.train(mode)
        self.value_head.train(mode)

    def parameters(self):
        """Return all trainable parameters."""
        return (
            list(self.trunk.parameters())
            + list(self.policy_head.parameters())
            + list(self.value_head.parameters())
        )

    def save_model(self, path: str) -> None:
        state = {
            'trunk': self.trunk.state_dict(),
            'policy_head': self.policy_head.state_dict(),
            'value_head': self.value_head.state_dict(),
        }
        torch.save(state, path)

    def load_model(self, path: str) -> None:
        state = torch.load(path, map_location='cpu')
        self.trunk.load_state_dict(state['trunk'])
        self.policy_head.load_state_dict(state['policy_head'])
        self.value_head.load_state_dict(state['value_head'])

    def encode_observation(self, obs: Observation, viewer_id: int = None) -> torch.Tensor:
        """Encode observation as 15-dim vector (same as value_based agent)."""
        if viewer_id is None:
            viewer_id = obs.current_player

        # 1. My hand (one-hot, 3 dims)
        hand_idx = self.CARD_MAP.get(obs.player_hand)
        hand_vec = torch.zeros(3)
        if hand_idx is not None:
            hand_vec[hand_idx] = 1.0

        # 2. Board card (one-hot, 4 dims): J, Q, K, None
        board_idx = self.CARD_MAP.get(obs.board, 3)
        board_vec = torch.zeros(4)
        board_vec[board_idx] = 1.0

        # 3. Pot (normalized, relative to viewer, 2 dims)
        p0_pot, p1_pot = obs.pot
        pot_rel = [p0_pot, p1_pot] if viewer_id == 0 else [p1_pot, p0_pot]
        pot_vec = torch.tensor(pot_rel) / self.MAX_CHIPS

        # 4-9. Scalar features (6 dims)
        features = torch.tensor([
            1.0 if viewer_id == obs.current_player else 0.0,
            float(viewer_id),
            float(obs.current_round),
            1.0 if obs.is_finished else 0.0,
            1.0 if (obs.board is not None and obs.player_hand == obs.board) else 0.0,
            obs.raises_this_round / 2.0,
        ])

        return torch.cat([hand_vec, board_vec, pot_vec, features]).unsqueeze(0)  # (1, 15)

    def forward(self, obs_enc: torch.Tensor):
        """Returns (logits, value) for a given encoded observation."""
        trunk_out = self.trunk(obs_enc)
        logits = self.policy_head(trunk_out)
        value = self.value_head(trunk_out)
        return logits, value

    def get_value(self, obs: Observation, viewer_id: int) -> torch.Tensor:
        """Returns scalar value estimate for a state."""
        enc = self.encode_observation(obs, viewer_id=viewer_id)
        with torch.no_grad():
            trunk_out = self.trunk(enc)
            value = self.value_head(trunk_out)
        return value.squeeze()

    def select_action(self, obs: Observation) -> Action:
        """Select action via masked softmax; sample in train mode, argmax in eval."""
        enc = self.encode_observation(obs, viewer_id=obs.current_player)
        logits, _ = self.forward(enc)
        logits = logits.squeeze(0)

        mask = torch.full((3,), -1e9)
        for a in obs.legal_actions:
            mask[a.value] = 0.0
        masked_logits = logits + mask
        probs = torch.softmax(masked_logits, dim=-1)

        if self.train_mode:
            dist = torch.distributions.Categorical(probs)
            action_idx = dist.sample().item()
        else:
            action_idx = probs.argmax().item()

        return Action(action_idx)
