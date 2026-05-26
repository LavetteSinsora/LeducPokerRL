"""
Bayesian Belief Agent for Leduc Hold'em.

Maintains an explicit belief distribution over the opponent's hand card,
updated within each hand based on observed actions. Uses a learned
likelihood model P(action | hand, game_state) to perform Bayesian updates.

Architecture:
  - Belief tracking: P(opponent_hand) initialized from card removal,
    then updated via Bayes' rule using a learned action likelihood model.
  - Likelihood model: Small MLP mapping (hand_onehot, game_features) -> P(action)
  - Value network: MLP(14 -> 64 -> 64 -> 1) with belief vector as input feature.
  - Action selection: 1-step lookahead (same as ValueBasedAgent).
"""

import numpy as np
import torch
import torch.nn as nn
from src.engine.leduc_game import Action, LeducGame
from src.engine.observation import Observation
from .base import BaseAgent
from . import belief_common


class LikelihoodModel(nn.Module):
    """
    Learns P(action | opponent_hand, game_state).

    Input: opponent_hand (3 one-hot) + game_state features (7 dims) = 10
      game_state features: board(4 one-hot incl None) + pot_ratio(1) + round(1) + raises(1)
    Output: 3-dim softmax over actions (FOLD, CALL, RAISE)
    """
    def __init__(self, input_size: int = 10, hidden_size: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 3),  # 3 actions
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns log-probabilities over actions."""
        return torch.log_softmax(self.net(x), dim=-1)


class BeliefValueNetwork(nn.Module):
    """
    Value network that takes belief-augmented state as input.

    Input (14 dims):
      hand(3 one-hot) + board(4 one-hot incl None) + pot(2 normalized)
      + belief(3) + round(1) + raises(1)
    Output: scalar V(s)
    """
    def __init__(self, input_size: int = 14, hidden_size: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class BeliefValueAgent(BaseAgent):
    """
    Bayesian Belief Agent that maintains and updates a belief distribution
    over the opponent's hand card during gameplay.

    The belief is initialized from card removal logic and updated after
    each opponent action using a learned likelihood model.
    """

    CARD_MAP = belief_common.CARD_MAP
    CARDS = belief_common.CARDS
    CARD_COUNTS = belief_common.CARD_COUNTS
    MAX_CHIPS = belief_common.MAX_CHIPS

    def __init__(self, model_path: str = None, temperature: float = 1.0):
        self.input_size = 14  # hand(3) + board(4) + pot(2) + belief(3) + round(1) + raises(1)
        self.temperature = temperature
        self.train_mode = False

        self.model = BeliefValueNetwork(self.input_size)
        self.likelihood_model = LikelihoodModel()

        if model_path:
            self.load_model(model_path)
        self.model.eval()
        self.likelihood_model.eval()

    def set_train_mode(self, mode: bool):
        self.train_mode = mode
        self.model.train(mode)
        self.likelihood_model.train(mode)

    def save_model(self, path: str) -> None:
        torch.save({
            'value_network': self.model.state_dict(),
            'likelihood_model': self.likelihood_model.state_dict(),
        }, path)

    def load_model(self, path: str) -> None:
        checkpoint = torch.load(path, weights_only=False)
        self.model.load_state_dict(checkpoint['value_network'])
        self.likelihood_model.load_state_dict(checkpoint['likelihood_model'])

    # ------------------------------------------------------------------
    # Belief computation
    # ------------------------------------------------------------------

    def initialize_belief(self, my_hand: str, board: str = None) -> np.ndarray:
        """
        Initialize P(opponent_hand) from card removal logic.

        Given my hand and the board card, compute the prior distribution
        over the opponent's possible hand cards.

        Returns:
            numpy array of shape (3,) with probabilities for [J, Q, K].
        """
        return belief_common.initialize_belief(my_hand, board)

    def _encode_likelihood_input(self, hand_idx: int, obs: Observation) -> torch.Tensor:
        """
        Encode input for the likelihood model: opponent_hand(3) + game_state(7).

        game_state features:
          board(4 one-hot incl None) + pot_ratio(1) + round(1) + raises(1)
        """
        # Opponent hand one-hot
        hand_vec = torch.zeros(3)
        hand_vec[hand_idx] = 1.0

        # Board card one-hot (4 dims: J, Q, K, None)
        board_idx = self.CARD_MAP.get(obs.board, 3)
        board_vec = torch.zeros(4)
        board_vec[board_idx] = 1.0

        # Pot ratio (total pot normalized)
        pot_total = sum(obs.pot)
        pot_ratio = torch.tensor([pot_total / self.MAX_CHIPS])

        # Round and raises
        features = torch.tensor([
            float(obs.current_round),
            obs.raises_this_round / 2.0,
        ])

        return torch.cat([hand_vec, board_vec, pot_ratio, features]).unsqueeze(0)

    def update_belief(self, belief: np.ndarray, action: Action,
                      obs: Observation) -> np.ndarray:
        """
        Bayesian update of belief given an observed opponent action.

        P(hand | action) proportional to P(action | hand) * P(hand)

        Args:
            belief: Current belief distribution, shape (3,).
            action: The action the opponent took.
            obs: The observation at the time the opponent acted.

        Returns:
            Updated belief distribution, shape (3,).
        """
        action_idx = int(action)
        likelihoods = np.zeros(3)

        with torch.no_grad():
            for hand_idx in range(3):
                if belief[hand_idx] < 1e-8:
                    continue
                inp = self._encode_likelihood_input(hand_idx, obs)
                log_probs = self.likelihood_model(inp)  # (1, 3)
                likelihoods[hand_idx] = torch.exp(log_probs[0, action_idx]).item()

        # Bayesian update
        posterior = belief * likelihoods
        total = posterior.sum()
        if total < 1e-10:
            return belief  # No update if everything is zero
        return posterior / total

    def compute_belief_from_history(self, obs: Observation) -> np.ndarray:
        """
        Compute the full belief vector by replaying the action history.

        Starts from the card-removal prior and applies Bayesian updates
        for each opponent action in the history.
        """
        def _update_callback(belief, action, running_state):
            hist_obs = Observation(
                player_hand=running_state['my_hand'],
                board=running_state['board'],
                pot=running_state['pot'],
                current_player=running_state['player_id'],
                current_round=running_state['current_round'],
                legal_actions=[],
                is_finished=False,
                raises_this_round=running_state['raises'],
            )
            return self.update_belief(belief, action, hist_obs)

        return belief_common.replay_belief_from_history(obs, _update_callback)

    # ------------------------------------------------------------------
    # Encoding
    # ------------------------------------------------------------------

    def encode_observation(self, obs: Observation, viewer_id: int = None,
                           belief: np.ndarray = None) -> torch.Tensor:
        """
        Encode observation with belief vector.

        Input vector (14 dims):
          hand(3) + board(4) + pot(2) + belief(3) + round(1) + raises(1)
        """
        if viewer_id is None:
            viewer_id = obs.current_player

        # 1. My hand (one-hot)
        hand_idx = self.CARD_MAP.get(obs.player_hand)
        hand_vec = torch.zeros(3)
        if hand_idx is not None:
            hand_vec[hand_idx] = 1.0

        # 2. Board card (one-hot): J, Q, K, None
        board_idx = self.CARD_MAP.get(obs.board, 3)
        board_vec = torch.zeros(4)
        board_vec[board_idx] = 1.0

        # 3. Pot (normalized, relative to viewer)
        p0_pot, p1_pot = obs.pot
        pot_rel = [p0_pot, p1_pot] if viewer_id == 0 else [p1_pot, p0_pot]
        pot_vec = torch.tensor(pot_rel, dtype=torch.float32) / self.MAX_CHIPS

        # 4. Belief vector (3 dims)
        if belief is None:
            belief = self.compute_belief_from_history(obs)
        belief_vec = torch.tensor(belief, dtype=torch.float32)

        # 5. Round and raises
        features = torch.tensor([
            float(obs.current_round),
            obs.raises_this_round / 2.0,
        ])

        return torch.cat([hand_vec, board_vec, pot_vec, belief_vec, features]).unsqueeze(0)

    # ------------------------------------------------------------------
    # Action selection (1-step lookahead)
    # ------------------------------------------------------------------

    def _get_value(self, obs: Observation, viewer_id: int,
                   belief: np.ndarray = None) -> float:
        encoded = self.encode_observation(obs, viewer_id=viewer_id, belief=belief)
        with torch.no_grad():
            return self.model(encoded).item()

    def get_action_evaluations(self, obs: Observation) -> list:
        """Run 1-step simulation and return predicted values with belief."""
        evaluations = []
        current_p = obs.current_player

        # Compute belief for the current state
        belief = self.compute_belief_from_history(obs)

        for action in obs.legal_actions:
            post_obs, done = LeducGame.simulate_action(obs, action)

            if done and action == Action.FOLD:
                val = -float(obs.pot[current_p])
            else:
                # Carry belief forward into simulated states
                # For the post-state after our own action, the belief
                # doesn't change (only opponent actions update it)
                val = self._get_value(post_obs, viewer_id=current_p, belief=belief)

            encoded = self.encode_observation(post_obs, viewer_id=current_p, belief=belief)
            evaluations.append({
                "action": action,
                "value": val,
                "is_terminal": done,
                "encoded": encoded,
                "belief": belief.copy(),
            })
        return evaluations

    def select_action(self, obs: Observation) -> Action:
        """Select action using 1-step lookahead with belief-augmented values."""
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
            print(f"Error in BeliefValueAgent selection: {e}")
            return results[0]["action"]
