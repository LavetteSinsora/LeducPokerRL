"""
DQN Agent — Deep Q-Network with target network and experience replay.

Q-network:      15 → 64 → 64 → 3 (one Q-value per action)
Target network: identical architecture, hard-copied every TARGET_UPDATE_FREQ episodes
Exploration:    epsilon-greedy (epsilon annealed 1.0 → 0.05 over 100K episodes)
"""

import copy

import torch
import torch.nn as nn

from agents.base import BaseAgent
from engine.leduc_game import Action
from engine.observation import Observation


class DQNAgent(BaseAgent):
    """
    Deep Q-Network with target network and experience replay.
    """

    CARD_MAP = {'J': 0, 'Q': 1, 'K': 2}
    MAX_CHIPS = 13

    def __init__(self, model_path: str = None, epsilon: float = 1.0):
        self.train_mode = False
        self.epsilon = epsilon

        self.q_net = nn.Sequential(
            nn.Linear(15, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 3),
        )
        self.target_net = copy.deepcopy(self.q_net)
        self.target_net.eval()  # Target net always in eval mode

        if model_path:
            self.load_model(model_path)
        self.q_net.eval()

    def set_train_mode(self, mode: bool):
        self.train_mode = mode
        self.q_net.train(mode)
        # target_net stays in eval mode always

    def update_target(self):
        """Hard copy q_net weights to target_net."""
        self.target_net.load_state_dict(self.q_net.state_dict())

    def save_model(self, path: str) -> None:
        torch.save(self.q_net.state_dict(), path)

    def load_model(self, path: str) -> None:
        self.q_net.load_state_dict(torch.load(path, map_location='cpu'))
        self.target_net.load_state_dict(self.q_net.state_dict())

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

    def select_action(self, obs: Observation) -> Action:
        """
        Epsilon-greedy in train mode, greedy in eval mode.
        Illegal actions are masked with -1e9 before argmax/sampling.
        """
        import random

        legal = obs.legal_actions
        if not legal:
            return Action.FOLD

        if self.train_mode and random.random() < self.epsilon:
            return random.choice(legal)

        enc = self.encode_observation(obs, viewer_id=obs.current_player)
        with torch.no_grad():
            q_vals = self.q_net(enc).squeeze(0)  # (3,)

        mask = torch.full((3,), -1e9)
        for a in legal:
            mask[a.value] = 0.0
        masked_q = q_vals + mask
        action_idx = masked_q.argmax().item()
        return Action(action_idx)
