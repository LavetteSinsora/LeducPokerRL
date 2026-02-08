import numpy as np
import torch
import torch.nn as nn
from src.engine.leduc_game import Action, LeducGame
from src.engine.observation import Observation
from .base import BaseAgent

class ValueNetwork(nn.Module):
    """
    A simple MLP to estimate action values.
    Input size depends on the encoding scheme.
    Output size: Number of possible actions (FOLD, CALL, RAISE).
    """
    def __init__(self, input_size, hidden_size=64):
        super(ValueNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1) # V(s) - Single state value
        )

    def forward(self, x):
        return self.net(x)

class ValueBasedAgent(BaseAgent):
    """
    A baseline RL agent that uses a value network to select actions.
    """
    def __init__(self, model_path=None):
        # Card mapping: J=0, Q=1, K=2
        self.card_map = {'J': 0, 'Q': 1, 'K': 2}
        
        # Input size calculation:
        # Private card (4: J, Q, K, UNKNOWN) + Board card (4) + Pot (2) + Round (1) + Current Player (2) + Terminal Marker (1)
        self.input_size = 4 + 4 + 2 + 1 + 2 + 1
        
        self.train_mode = False
        self.model = ValueNetwork(self.input_size)
        if model_path:
            self.model.load_state_dict(torch.load(model_path))
        self.model.eval()

    def set_train_mode(self, mode: bool):
        self.train_mode = mode
        if mode:
            self.model.train()
        else:
            self.model.eval()

    def encode_observation(self, obs: Observation):
        """
        Encodes the observation into a flat vector for the ValueNetwork.
        """
        # 1. Private card (one-hot)
        # J=0, Q=1, K=2, UNKNOWN=3
        private_card = obs.player_hand
        private_one_hot = np.zeros(4)
        if private_card == 'UNKNOWN':
            private_one_hot[3] = 1
        else:
            private_card_idx = self.card_map[private_card]
            private_one_hot[private_card_idx] = 1
        
        # 2. Board card (one-hot, with extra index for None)
        board_one_hot = np.zeros(4)
        if obs.board is None:
            board_one_hot[3] = 1
        else:
            board_card_idx = self.card_map[obs.board]
            board_one_hot[board_card_idx] = 1
            
        # 3. Pot (normalized by some constant, e.g., max expected chips)
        max_chips = 20 # Arbitrary normalization
        pot = np.array(obs.pot) / max_chips
        
        # 4. Round (normalized)
        round_idx = np.array([obs.current_round])
        
        # 5. Current Player (one-hot)
        player_one_hot = np.zeros(2)
        player_one_hot[obs.current_player] = 1
        
        # 6. Terminal Marker (1 if finished, 0 otherwise)
        terminal_marker = np.array([1.0 if obs.is_finished else 0.0])
        
        # Concatenate everything
        feature_vector = np.concatenate([
            private_one_hot,
            board_one_hot,
            pot,
            round_idx,
            player_one_hot,
            terminal_marker
        ])
        
        return torch.FloatTensor(feature_vector).unsqueeze(0) # Add batch dimension

    def _evaluate_state(self, obs: Observation):
        """
        Encodes observation and returns (predicted_value, encoded_vector).
        We always use no_grad here because selection is an inference-only step.
        """
        encoded = self.encode_observation(obs)
        with torch.no_grad():
            val = self.model(encoded).item()
        return val, encoded

    def select_action(self, obs: Observation) -> Action:
        """
        Selects the best action based on 1-step simulation and value network.
        If in train_mode, returns (action, encoded_state) for the trainer.
        """
        simulator = LeducGame()
        results = []

        for action in obs.legal_actions:
            # Simulation step
            simulator.set_state(obs)
            _, _, done, _ = simulator.step(action)
            
            if done:
                if action == Action.FOLD:
                    # Case 1: Fold (Deterministic payout)
                    val = -float(obs.pot[obs.current_player])
                    encoded = None
                else:
                    # Case 2: Showdown (Value network on resulting terminal state)
                    terminal_obs = Observation(
                        player_hand=obs.player_hand,
                        board=simulator.board,
                        pot=list(simulator.pot),
                        current_player=obs.current_player,
                        current_round=simulator.current_round,
                        legal_actions=[],
                        is_finished=True
                    )
                    val, encoded = self._evaluate_state(terminal_obs)
            else:
                # Case 3: Mid-game transition
                # View next state from our perspective
                next_obs = simulator.get_observation(viewer_id=obs.current_player)
                v_model, encoded = self._evaluate_state(next_obs)
                
                # Zero-sum normalization: 
                # If it's the opponent's turn, the network predicts their value.
                val = v_model if next_obs.current_player == obs.current_player else -v_model
            
            results.append((action, val, encoded))

        # Greedy selection over simulated futures
        best_action, _, best_encoded = max(results, key=lambda x: x[1])

        if self.train_mode:
            return best_action, best_encoded
        return best_action
