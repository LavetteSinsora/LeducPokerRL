"""ValueDimAgent: value network with configurable hidden dimensions.

Used by value_dim_search_v1 to probe the minimum viable network capacity
for learning a strong Leduc Hold'em strategy.
"""

import torch
import torch.nn as nn
from typing import List

from agents.base import BaseAgent
from engine.leduc_game import Action, LeducGame
from engine.observation import Observation


class ValueDimAgent(BaseAgent):
    """
    Value-based agent with a dynamically-sized MLP.

    The network architecture is: 15 -> hidden_dims[0] -> ... -> hidden_dims[-1] -> 1
    All hidden layers use ReLU activations.

    Args:
        hidden_dims: List of hidden layer sizes, e.g. [32, 32] or [32, 16].
        temperature: Boltzmann exploration temperature used during training.
    """

    # Constants for encoding — identical to value_based agent
    CARD_MAP = {'J': 0, 'Q': 1, 'K': 2}
    MAX_CHIPS = 13
    INPUT_SIZE = 15  # 3 + 4 + 2 + 1 + 1 + 1 + 1 + 1 + 1

    def __init__(self, hidden_dims: List[int] = None, temperature: float = 1.0):
        if hidden_dims is None:
            hidden_dims = [32, 32]

        self.hidden_dims = hidden_dims
        self.temperature = temperature
        self.train_mode = False

        # Build sequential network dynamically
        layers = []
        in_dim = self.INPUT_SIZE
        for h in hidden_dims:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        self.model = nn.Sequential(*layers)

        self.model.eval()

    def set_train_mode(self, mode: bool):
        self.train_mode = mode
        self.model.train(mode)

    def save_model(self, path: str) -> None:
        """Save model weights to disk."""
        torch.save(self.model.state_dict(), path)

    def load_model(self, path: str) -> None:
        """Load model weights from disk."""
        self.model.load_state_dict(torch.load(path))

    def encode_observation(self, obs: Observation, viewer_id: int = None) -> torch.Tensor:
        """
        Encodes the observation relative to the viewer_id.
        Identical encoding to value_based agent (15-dim).
        """
        if viewer_id is None:
            viewer_id = obs.current_player

        # 1. My hand (one-hot)
        hand_idx = self.CARD_MAP.get(obs.player_hand)
        hand_vec = torch.zeros(3)
        if hand_idx is not None:
            hand_vec[hand_idx] = 1.0

        # 2. Board card (one-hot): J, Q, K, None (index 3 is None)
        board_idx = self.CARD_MAP.get(obs.board, 3)
        board_vec = torch.zeros(4)
        board_vec[board_idx] = 1.0

        # 3. Pot (normalized, relative to viewer)
        p0_pot, p1_pot = obs.pot
        pot_rel = [p0_pot, p1_pot] if viewer_id == 0 else [p1_pot, p0_pot]
        pot_vec = torch.tensor(pot_rel) / self.MAX_CHIPS

        # 4-9. Scalar features
        features = torch.tensor([
            1.0 if viewer_id == obs.current_player else 0.0,  # My turn
            float(viewer_id),                                  # Position
            float(obs.current_round),                          # Round
            1.0 if obs.is_finished else 0.0,                   # Terminal
            1.0 if (obs.board is not None and obs.player_hand == obs.board) else 0.0,  # Has Pair
            obs.raises_this_round / 2.0,                       # Raises normalized
        ])

        return torch.cat([hand_vec, board_vec, pot_vec, features]).unsqueeze(0)

    def _get_value(self, obs: Observation, viewer_id: int) -> float:
        """Predicts the value for a given player using the network."""
        encoded = self.encode_observation(obs, viewer_id=viewer_id)
        with torch.no_grad():
            return self.model(encoded).item()

    def get_action_evaluations(self, obs: Observation) -> list:
        """Runs 1-step simulation and returns predicted values."""
        evaluations = []
        current_p = obs.current_player

        for action in obs.legal_actions:
            post_obs, done = LeducGame.simulate_action(obs, action)

            if done and action == Action.FOLD:
                val = -float(obs.pot[current_p])
            else:
                val = self._get_value(post_obs, viewer_id=current_p)

            encoded = self.encode_observation(post_obs, viewer_id=current_p)
            evaluations.append({
                "action": action,
                "value": val,
                "is_terminal": done,
                "encoded": encoded,
            })
        return evaluations

    def select_action(self, obs: Observation) -> Action:
        """Selects an action based on 1-step simulation and value network."""
        results = self.get_action_evaluations(obs)

        if not results:
            return Action.FOLD

        try:
            if self.train_mode:
                # Softmax (Boltzmann) exploration
                values = torch.tensor([r["value"] for r in results])
                probs = torch.softmax(values / self.temperature, dim=0)
                idx = torch.multinomial(probs, 1).item()
                return results[idx]["action"]
            else:
                # Greedy selection
                return max(results, key=lambda x: x["value"])["action"]
        except Exception as e:
            print(f"Error in ValueDimAgent selection: {e}")
            return results[0]["action"]
