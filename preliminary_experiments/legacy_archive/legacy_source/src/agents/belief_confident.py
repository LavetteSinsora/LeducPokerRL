"""
Belief Confident Agent for Leduc Hold'em.

Extends BeliefValueAgent with a confidence dimension that tells the value
network how much to trust the belief estimate.

Architecture:
  - Same belief tracking as BeliefValueAgent.
  - Value network: MLP(15 -> 64 -> 64 -> 1) — one extra dim for confidence.
  - Input: hand(3) + board(4) + pot(2) + belief(3) + round(1) + raises(1) + confidence(1) = 15
  - Confidence = min(n_games_observed, 30) / 30, where n_games_observed counts
    how many games have been played against the current opponent in this session.

The value network can learn to:
  - When confidence ~ 0: ignore belief, play a base strategy
  - When confidence ~ 1: trust belief, adapt strategy accordingly
"""

import numpy as np
import torch
from src.engine.leduc_game import Action, LeducGame
from src.engine.observation import Observation
from .belief_value import BeliefValueAgent, BeliefValueNetwork


class BeliefConfidentAgent(BeliefValueAgent):
    """
    Bayesian Belief Agent augmented with a confidence score indicating
    how reliable the current belief estimate is.

    The confidence is derived from the number of games observed against
    the current opponent in the session: min(n_games, 30) / 30.
    """

    CONFIDENCE_CAP = 30  # Cap for normalization

    def __init__(self, model_path: str = None, temperature: float = 1.0):
        # Skip BeliefValueAgent.__init__ to set custom input_size
        self.input_size = 15  # hand(3) + board(4) + pot(2) + belief(3) + round(1) + raises(1) + confidence(1)
        self.temperature = temperature
        self.train_mode = False

        self.model = BeliefValueNetwork(self.input_size)
        from .belief_value import LikelihoodModel
        self.likelihood_model = LikelihoodModel()

        # Session game counter (tracks games played against current opponent)
        self._n_games_observed = 0

        if model_path:
            self.load_model(model_path)
        self.model.eval()
        self.likelihood_model.eval()

    @property
    def confidence(self) -> float:
        """Current confidence score in [0, 1]."""
        return min(self._n_games_observed, self.CONFIDENCE_CAP) / self.CONFIDENCE_CAP

    def reset_session(self):
        """Reset the game counter for a new session/opponent."""
        self._n_games_observed = 0

    def increment_game_count(self):
        """Increment the game counter after a completed game."""
        self._n_games_observed += 1

    def set_game_count(self, n: int):
        """Set the game counter directly (for training/testing)."""
        self._n_games_observed = n

    def encode_observation(self, obs: Observation, viewer_id: int = None,
                           belief: np.ndarray = None,
                           confidence: float = None) -> torch.Tensor:
        """
        Encode observation with belief vector and confidence score.

        Input vector (15 dims):
          hand(3) + board(4) + pot(2) + belief(3) + round(1) + raises(1) + confidence(1)
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

        # 6. Confidence score
        if confidence is None:
            confidence = self.confidence
        conf_vec = torch.tensor([confidence], dtype=torch.float32)

        return torch.cat([hand_vec, board_vec, pot_vec, belief_vec, features, conf_vec]).unsqueeze(0)

    def _get_value(self, obs: Observation, viewer_id: int,
                   belief: np.ndarray = None,
                   confidence: float = None) -> float:
        encoded = self.encode_observation(obs, viewer_id=viewer_id,
                                          belief=belief, confidence=confidence)
        with torch.no_grad():
            return self.model(encoded).item()

    def get_action_evaluations(self, obs: Observation) -> list:
        """Run 1-step simulation and return predicted values with belief + confidence."""
        evaluations = []
        current_p = obs.current_player

        # Compute belief for the current state
        belief = self.compute_belief_from_history(obs)
        conf = self.confidence

        for action in obs.legal_actions:
            post_obs, done = LeducGame.simulate_action(obs, action)

            if done and action == Action.FOLD:
                val = -float(obs.pot[current_p])
            else:
                val = self._get_value(post_obs, viewer_id=current_p,
                                      belief=belief, confidence=conf)

            encoded = self.encode_observation(post_obs, viewer_id=current_p,
                                              belief=belief, confidence=conf)
            evaluations.append({
                "action": action,
                "value": val,
                "is_terminal": done,
                "encoded": encoded,
                "belief": belief.copy(),
                "confidence": conf,
            })
        return evaluations

    def select_action(self, obs: Observation) -> Action:
        """Select action using 1-step lookahead with belief + confidence."""
        results = self.get_action_evaluations(obs)

        if not results:
            return Action.FOLD

        try:
            if self.train_mode:
                values = torch.tensor([r["value"] for r in results])
                probs = torch.softmax(values / self.temperature, dim=0)
                idx = torch.multinomial(probs, 1).item()
                return results[idx]["action"]
            else:
                return max(results, key=lambda x: x["value"])["action"]
        except Exception as e:
            print(f"Error in BeliefConfidentAgent selection: {e}")
            return results[0]["action"]
