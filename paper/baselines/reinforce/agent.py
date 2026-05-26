"""
REINFORCE Agent — Policy Gradient (Monte Carlo PG).

Architecture: 15 → 64 → 64 → 3 (action logits)
Learns a stochastic action distribution directly.
No value function — uses full-episode returns for credit assignment.
"""

import torch
import torch.nn as nn

from agents.base import BaseAgent
from engine.leduc_game import Action
from engine.observation import Observation


class REINFORCEAgent(BaseAgent):
    """
    Policy gradient agent (REINFORCE / Monte Carlo PG).
    Architecture: 15 → 64 → 64 → 3 (action logits)
    Learns a stochastic action distribution directly.
    No value function — uses full-episode returns for credit assignment.
    """

    CARD_MAP = {'J': 0, 'Q': 1, 'K': 2}
    MAX_CHIPS = 13

    def __init__(self, model_path: str = None):
        self.train_mode = False
        self.policy_net = nn.Sequential(
            nn.Linear(15, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 3),
        )
        if model_path:
            self.load_model(model_path)
        self.policy_net.eval()

    def set_train_mode(self, mode: bool):
        self.train_mode = mode
        self.policy_net.train(mode)

    def save_model(self, path: str) -> None:
        torch.save(self.policy_net.state_dict(), path)

    def load_model(self, path: str) -> None:
        self.policy_net.load_state_dict(torch.load(path, map_location='cpu'))

    def encode_observation(self, obs: Observation, viewer_id: int = None) -> torch.Tensor:
        """Encode observation as 15-dim vector (same as value_based agent)."""
        if viewer_id is None:
            viewer_id = obs.current_player

        # 1. My hand (one-hot, 3 dims)
        hand_idx = self.CARD_MAP.get(obs.player_hand)
        hand_vec = torch.zeros(3)
        if hand_idx is not None:
            hand_vec[hand_idx] = 1.0

        # 2. Board card (one-hot, 4 dims): J, Q, K, None (index 3 = no board)
        board_idx = self.CARD_MAP.get(obs.board, 3)
        board_vec = torch.zeros(4)
        board_vec[board_idx] = 1.0

        # 3. Pot (normalized, relative to viewer, 2 dims)
        p0_pot, p1_pot = obs.pot
        pot_rel = [p0_pot, p1_pot] if viewer_id == 0 else [p1_pot, p0_pot]
        pot_vec = torch.tensor(pot_rel) / self.MAX_CHIPS

        # 4-9. Scalar features (6 dims)
        features = torch.tensor([
            1.0 if viewer_id == obs.current_player else 0.0,  # My turn
            float(viewer_id),                                   # Position
            float(obs.current_round),                           # Round
            1.0 if obs.is_finished else 0.0,                    # Terminal
            1.0 if (obs.board is not None and obs.player_hand == obs.board) else 0.0,  # Has Pair
            obs.raises_this_round / 2.0,                        # Raises normalized
        ])

        return torch.cat([hand_vec, board_vec, pot_vec, features]).unsqueeze(0)  # (1, 15)

    def select_action(self, obs: Observation) -> Action:
        """Select action via masked softmax; sample in train mode, argmax in eval."""
        enc = self.encode_observation(obs, viewer_id=obs.current_player)
        logits = self.policy_net(enc).squeeze(0)  # (3,)

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
