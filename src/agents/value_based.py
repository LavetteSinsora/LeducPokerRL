import numpy as np
import torch
import torch.nn as nn
from src.engine.leduc_game import Action, LeducGame
from src.engine.observation import Observation
from .base import BaseAgent

class ValueNetwork(nn.Module):
    """
    A simple MLP to estimate state values.
    Input size depends on the encoding scheme.
    Output size: 1 (V(s) - Single state value).
    """
    def __init__(self, input_size: int, hidden_size: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class ValueBasedAgent(BaseAgent):
    """
    A baseline RL agent that uses a value network to select actions.
    Uses softmax (Boltzmann) exploration during training to prevent reward hacking.
    """
    
    # Constants for encoding
    CARD_MAP = {'J': 0, 'Q': 1, 'K': 2}
    MAX_CHIPS = 13
    
    def __init__(self, model_path: str = None, temperature: float = 1.0):
        # Input size: hand (3) + board (4) + pot (2) + turn (1) + pos (1) + round (1) + terminal (1) + pair (1) + raises (1)
        self.input_size = 3 + 4 + 2 + 1 + 1 + 1 + 1 + 1 + 1
        self.temperature = temperature
        self.train_mode = False

        self.model = ValueNetwork(self.input_size)
        if model_path:
            self.load_model(model_path)
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
        
        # 4-9. Scalable features
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
            print(f"Error in ValueBasedAgent selection: {e}")
            return results[0]["action"]


