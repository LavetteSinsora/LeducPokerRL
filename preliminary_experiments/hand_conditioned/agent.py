"""Hand-conditioned opponent action model and belief update utilities."""

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from engine.leduc_game import Action
from engine.observation import Observation


CARD_MAP = {"J": 0, "Q": 1, "K": 2}
CARDS = ["J", "Q", "K"]
CARD_COUNTS = [2, 2, 2]
MAX_CHIPS = 13
DEFAULT_STATS = [0.5, 0.5, 0.5, 0.0]


def initialize_belief(viewer_hand: str) -> np.ndarray:
    """Card-removal prior before seeing the board."""
    counts = list(CARD_COUNTS)
    counts[CARD_MAP[viewer_hand]] -= 1
    total = sum(counts)
    return np.array(counts, dtype=np.float32) / total


def update_belief_with_board(belief: np.ndarray, viewer_hand: str, board: str) -> np.ndarray:
    """Bayesian board-reveal update: posterior(h) ∝ prior(h) * P(board | h, viewer_hand)."""
    viewer_idx = CARD_MAP[viewer_hand]
    board_idx = CARD_MAP[board]
    likelihoods = np.zeros(3, dtype=np.float32)

    for hand_idx in range(3):
        counts = list(CARD_COUNTS)
        counts[viewer_idx] -= 1
        counts[hand_idx] -= 1
        remaining_total = sum(counts)
        if remaining_total <= 0 or counts[board_idx] <= 0:
            likelihoods[hand_idx] = 0.0
        else:
            likelihoods[hand_idx] = counts[board_idx] / remaining_total

    posterior = belief * likelihoods
    total = posterior.sum()
    if total <= 1e-10:
        return belief
    return posterior / total


class HandConditionedActionLikelihood(nn.Module):
    """Predict P(action | public_state, session_stats, candidate_hand)."""

    def __init__(self, input_size: int = 16, hidden_size: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 3),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.log_softmax(self.net(x), dim=-1)


class HandConditionedActionModel:
    """Wrapper around the likelihood network plus belief update helpers."""

    def __init__(self, model_path: str = None, hidden_size: int = 64):
        self.hidden_size = hidden_size
        self.model = HandConditionedActionLikelihood(hidden_size=hidden_size)
        if model_path:
            self.load_model(model_path)
        self.model.eval()

    def set_train_mode(self, mode: bool):
        self.model.train(mode)

    def save_model(self, path: str) -> None:
        torch.save({"likelihood_model": self.model.state_dict(), "hidden_size": self.hidden_size}, path)

    def load_model(self, path: str) -> None:
        checkpoint = torch.load(path, weights_only=False)
        state = checkpoint["likelihood_model"] if "likelihood_model" in checkpoint else checkpoint
        self.model.load_state_dict(state)

    @staticmethod
    def stats_to_tensor(obs: Observation) -> torch.Tensor:
        if obs.opponent_stats is not None and hasattr(obs.opponent_stats, "to_feature_vector"):
            return torch.tensor(obs.opponent_stats.to_feature_vector(), dtype=torch.float32)
        return torch.tensor(DEFAULT_STATS, dtype=torch.float32)

    @staticmethod
    def hand_to_tensor(hand: str) -> torch.Tensor:
        vec = torch.zeros(3, dtype=torch.float32)
        vec[CARD_MAP[hand]] = 1.0
        return vec

    @staticmethod
    def encode_public_state(obs: Observation, viewer_id: int) -> torch.Tensor:
        board_vec = torch.zeros(4, dtype=torch.float32)
        board_vec[CARD_MAP[obs.board] if obs.board is not None else 3] = 1.0

        p0_pot, p1_pot = obs.pot
        pot_rel = [p0_pot, p1_pot] if viewer_id == 0 else [p1_pot, p0_pot]
        pot_vec = torch.tensor(pot_rel, dtype=torch.float32) / MAX_CHIPS

        facing_raise = 1.0 if pot_rel[1] > pot_rel[0] else 0.0
        features = torch.tensor(
            [float(obs.current_round), obs.raises_this_round / 2.0, facing_raise],
            dtype=torch.float32,
        )
        return torch.cat([board_vec, pot_vec, features], dim=0)

    def encode_example(self, obs: Observation, viewer_id: int, candidate_hand: str) -> torch.Tensor:
        public_state = self.encode_public_state(obs, viewer_id)
        stats = self.stats_to_tensor(obs)
        hand_vec = self.hand_to_tensor(candidate_hand)
        return torch.cat([public_state, stats, hand_vec], dim=0).unsqueeze(0)

    def predict_log_probs(self, obs: Observation, viewer_id: int, candidate_hand: str) -> torch.Tensor:
        encoded = self.encode_example(obs, viewer_id, candidate_hand)
        return self.model(encoded)

    def predict_action_probs(self, obs: Observation, viewer_id: int, candidate_hand: str) -> torch.Tensor:
        return torch.exp(self.predict_log_probs(obs, viewer_id, candidate_hand))

    def update_belief(
        self,
        belief: np.ndarray,
        obs: Observation,
        viewer_id: int,
        observed_action: Action,
    ) -> np.ndarray:
        """Bayesian update with learned action likelihoods."""
        action_idx = int(observed_action)
        likelihoods = np.zeros(3, dtype=np.float32)

        with torch.no_grad():
            for hand in CARDS:
                hand_idx = CARD_MAP[hand]
                if belief[hand_idx] <= 1e-8:
                    continue
                probs = self.predict_action_probs(obs, viewer_id, hand).squeeze(0)
                likelihoods[hand_idx] = probs[action_idx].item()

        posterior = belief * likelihoods
        total = posterior.sum()
        if total <= 1e-10:
            return belief
        return posterior / total

    def belief_true_hand_probability(self, belief: np.ndarray, true_hand: str) -> float:
        return float(belief[CARD_MAP[true_hand]])

    def belief_top1_correct(self, belief: np.ndarray, true_hand: str) -> float:
        return float(int(int(np.argmax(belief)) == CARD_MAP[true_hand]))

    def majority_hand_baseline(self, viewer_hand: str, board: str = None) -> np.ndarray:
        prior = initialize_belief(viewer_hand)
        if board is not None:
            prior = update_belief_with_board(prior, viewer_hand, board)
        return prior

    @staticmethod
    def tvd(p: np.ndarray, q: np.ndarray) -> float:
        return float(0.5 * np.abs(p - q).sum())
