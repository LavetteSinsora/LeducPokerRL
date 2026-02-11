import torch
import torch.nn as nn
from src.engine.leduc_game import Action
from src.engine.observation import Observation
from .base import BaseAgent


class PolicyNetwork(nn.Module):
    """Maps a game state to action probabilities."""
    def __init__(self, input_size: int, hidden_size: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 3),  # 3 actions: fold, call, raise
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.net(x)
        return torch.softmax(logits, dim=-1)


class PolicyGradientAgent(BaseAgent):
    """
    A policy gradient agent that directly learns action probabilities.

    During training, it samples actions from its learned distribution
    (so it explores different strategies). During evaluation, it picks
    the highest-probability action (greedy).
    """

    CARD_MAP = {'J': 0, 'Q': 1, 'K': 2}
    MAX_CHIPS = 13

    def __init__(self, model_path: str = None):
        self.input_size = 15  # 3 (hand) + 4 (board) + 2 (pot) + 6 (features)
        self.train_mode = False

        self.model = PolicyNetwork(self.input_size)
        if model_path:
            self.load_model(model_path)
        self.model.eval()

    def encode_observation(self, obs: Observation, **kwargs) -> torch.Tensor:
        """Turn a game observation into a tensor the network can process."""
        # One-hot encode the player's hand card (J/Q/K)
        hand_vec = torch.zeros(3)
        hand_idx = self.CARD_MAP.get(obs.player_hand)
        if hand_idx is not None:
            hand_vec[hand_idx] = 1.0

        # One-hot encode the board card (J/Q/K/None)
        board_vec = torch.zeros(4)
        board_idx = self.CARD_MAP.get(obs.board, 3)  # 3 = no board card yet
        board_vec[board_idx] = 1.0

        # Pot sizes, normalized
        pot_vec = torch.tensor(obs.pot, dtype=torch.float32) / self.MAX_CHIPS

        # Extra features
        features = torch.tensor([
            float(obs.current_player),
            float(obs.current_round),
            1.0 if obs.is_finished else 0.0,
            1.0 if (obs.board and obs.player_hand == obs.board) else 0.0,  # pair?
            obs.raises_this_round / 2.0,
            1.0 if Action.RAISE in obs.legal_actions else 0.0,  # can raise?
        ])

        return torch.cat([hand_vec, board_vec, pot_vec, features]).unsqueeze(0)

    def select_action(self, obs: Observation) -> Action:
        """Pick an action based on the policy network's output."""
        encoded = self.encode_observation(obs)

        with torch.no_grad():
            probs = self.model(encoded).squeeze(0)  # [P(fold), P(call), P(raise)]

        # Mask out illegal actions (set their probability to 0)
        legal_mask = torch.zeros(3)
        for action in obs.legal_actions:
            legal_mask[action.value] = 1.0
        probs = probs * legal_mask

        # Re-normalize so probabilities sum to 1
        probs = probs / probs.sum()

        if self.train_mode:
            # Sample from the distribution (exploration)
            action_idx = torch.multinomial(probs, 1).item()
        else:
            # Pick the most probable action (greedy)
            action_idx = probs.argmax().item()

        return Action(action_idx)

    def set_train_mode(self, mode: bool):
        self.train_mode = mode
        self.model.train(mode)

    def save_model(self, path: str):
        torch.save(self.model.state_dict(), path)

    def load_model(self, path: str):
        self.model.load_state_dict(torch.load(path))
