import torch
import torch.nn as nn
from src.engine.leduc_game import Action
from src.engine.observation import Observation
from .base import BaseAgent


class ActorCriticNetwork(nn.Module):
    """
    Actor-Critic network with a shared backbone.

    Architecture:
      Shared backbone: Linear(15->64) -> ReLU -> Linear(64->64) -> ReLU
      Policy head (actor):  Linear(64->3) -> Softmax  (action probabilities)
      Value head (critic):  Linear(64->1)              (scalar state value)

    The shared backbone means both heads benefit from each other's gradients,
    leading to more sample-efficient learning than separate networks.
    """

    def __init__(self, input_size: int, hidden_size: int = 64):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )
        self.policy_head = nn.Linear(hidden_size, 3)  # 3 actions: fold, call, raise
        self.value_head = nn.Linear(hidden_size, 1)    # scalar state value

    def forward(self, x: torch.Tensor):
        """
        Returns:
            probs: action probabilities, shape (batch, 3)
            value: state value estimate, shape (batch, 1)
        """
        features = self.backbone(x)
        probs = torch.softmax(self.policy_head(features), dim=-1)
        value = self.value_head(features)
        return probs, value


class ActorCriticAgent(BaseAgent):
    """
    An actor-critic agent that learns both a policy and a value function.

    This is an incremental improvement over PolicyGradientAgent: the value
    head provides a learned baseline V(s) that reduces variance in the
    policy gradient signal. Instead of reinforcing with raw reward R,
    we use advantage = R - V(s), which tells the agent whether the outcome
    was better or worse than expected.

    During training, it samples actions from its learned distribution.
    During evaluation, it picks the highest-probability legal action.
    """

    CARD_MAP = {'J': 0, 'Q': 1, 'K': 2}
    MAX_CHIPS = 13

    def __init__(self, model_path: str = None):
        self.input_size = 15  # 3 (hand) + 4 (board) + 2 (pot) + 6 (features)
        self.train_mode = False

        self.model = ActorCriticNetwork(self.input_size)
        if model_path:
            self.load_model(model_path)
        self.model.eval()

    def encode_observation(self, obs: Observation, **kwargs) -> torch.Tensor:
        """Turn a game observation into a tensor the network can process.

        Same 15-dim encoding as PolicyGradientAgent:
          [hand_onehot(3), board_onehot(4), pot_norm(2), features(6)]
        """
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
        """Pick an action based on the policy head's output."""
        encoded = self.encode_observation(obs)

        with torch.no_grad():
            probs, _ = self.model(encoded)
            probs = probs.squeeze(0)  # [P(fold), P(call), P(raise)]

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

    def get_action_evaluations(self, obs: Observation) -> list:
        """Returns per-action probabilities AND value estimate for the analyzer UI."""
        encoded = self.encode_observation(obs)
        with torch.no_grad():
            probs, value = self.model(encoded)
            probs = probs.squeeze(0)   # raw [P(fold), P(call), P(raise)]
            value = value.squeeze(0).item()  # scalar V(s)

        # Legal masking + renormalization
        legal_mask = torch.zeros(3)
        for action in obs.legal_actions:
            legal_mask[action.value] = 1.0
        masked = probs * legal_mask
        masked = masked / masked.sum()

        return [
            {
                "action": a,
                "probability": masked[a.value].item(),
                "raw_probability": probs[a.value].item(),
                "value_estimate": value,
                "encoded": encoded,
            }
            for a in obs.legal_actions
        ]

    def set_train_mode(self, mode: bool):
        self.train_mode = mode
        self.model.train(mode)

    def save_model(self, path: str):
        torch.save(self.model.state_dict(), path)

    def load_model(self, path: str):
        self.model.load_state_dict(torch.load(path))
