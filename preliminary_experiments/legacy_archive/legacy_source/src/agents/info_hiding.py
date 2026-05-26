"""
Information-Hiding Agent for Leduc Hold'em.

An actor-critic agent with an auxiliary "spy" network that tries to predict the
agent's hand from its action history.  The main policy is trained adversarially
to make the spy FAIL, encouraging deceptive / mixed-strategy play.

Architecture:
  - Policy head: MLP(15 -> 64 -> 64 -> 3) with softmax
  - Value head:  MLP(15 -> 64 -> 64 -> 1)
  - Spy network:  MLP(20 -> 32 -> 3) classifier (separate parameters)

The spy is trained to predict which card (J/Q/K) the agent holds from the
sequence of actions taken during the hand.  The policy gradient loss is then
augmented with a NEGATIVE spy-accuracy term, pushing the policy toward actions
that make the spy's job harder.
"""

import torch
import torch.nn as nn
from src.engine.leduc_game import Action
from src.engine.observation import Observation
from .base import BaseAgent


class ActorCriticNetwork(nn.Module):
    """Shared-backbone actor-critic with policy and value heads."""

    def __init__(self, input_size: int, hidden_size: int = 64):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )
        self.policy_head = nn.Linear(hidden_size, 3)   # fold / call / raise
        self.value_head = nn.Linear(hidden_size, 1)     # scalar V(s)

    def forward(self, x: torch.Tensor):
        features = self.backbone(x)
        probs = torch.softmax(self.policy_head(features), dim=-1)
        value = self.value_head(features)
        return probs, value


class SpyNetwork(nn.Module):
    """
    Classifier that predicts the agent's hole card from its action sequence.

    Input (20-dim):
        4 action slots x 4 one-hot features = 16
        + pot_self (1) + pot_opp (1) + round_reached (1) + num_raises (1) = 4

    Output:
        P(J), P(Q), P(K)  — softmax over 3 cards.
    """

    def __init__(self, input_size: int = 20, hidden_size: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 3),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)   # raw logits; apply softmax / cross-entropy externally


class InfoHidingAgent(BaseAgent):
    """
    Information-Hiding Actor-Critic agent.

    Identical interface to ActorCriticAgent, but carries an additional SpyNetwork
    whose weights are only touched by InfoHidingTrainer.
    """

    CARD_MAP = {'J': 0, 'Q': 1, 'K': 2}
    MAX_CHIPS = 13

    def __init__(self, model_path: str = None):
        self.input_size = 15   # 3 (hand) + 4 (board) + 2 (pot) + 6 (features)
        self.train_mode = False

        self.model = ActorCriticNetwork(self.input_size)
        self.spy = SpyNetwork(input_size=20, hidden_size=32)

        if model_path:
            self.load_model(model_path)
        self.model.eval()
        self.spy.eval()

    # ── observation encoding (same 15-dim as ActorCriticAgent) ──────

    def encode_observation(self, obs: Observation, **kwargs) -> torch.Tensor:
        hand_vec = torch.zeros(3)
        hand_idx = self.CARD_MAP.get(obs.player_hand)
        if hand_idx is not None:
            hand_vec[hand_idx] = 1.0

        board_vec = torch.zeros(4)
        board_idx = self.CARD_MAP.get(obs.board, 3)
        board_vec[board_idx] = 1.0

        pot_vec = torch.tensor(obs.pot, dtype=torch.float32) / self.MAX_CHIPS

        features = torch.tensor([
            float(obs.current_player),
            float(obs.current_round),
            1.0 if obs.is_finished else 0.0,
            1.0 if (obs.board and obs.player_hand == obs.board) else 0.0,
            obs.raises_this_round / 2.0,
            1.0 if Action.RAISE in obs.legal_actions else 0.0,
        ])

        return torch.cat([hand_vec, board_vec, pot_vec, features]).unsqueeze(0)

    # ── action selection ────────────────────────────────────────────

    def select_action(self, obs: Observation) -> Action:
        encoded = self.encode_observation(obs)

        with torch.no_grad():
            probs, _ = self.model(encoded)
            probs = probs.squeeze(0)

        legal_mask = torch.zeros(3)
        for action in obs.legal_actions:
            legal_mask[action.value] = 1.0
        probs = probs * legal_mask
        probs = probs / probs.sum()

        if self.train_mode:
            action_idx = torch.multinomial(probs, 1).item()
        else:
            action_idx = probs.argmax().item()

        return Action(action_idx)

    # ── evaluations for analyzer UI ─────────────────────────────────

    def get_action_evaluations(self, obs: Observation) -> list:
        encoded = self.encode_observation(obs)
        with torch.no_grad():
            probs, value = self.model(encoded)
            probs = probs.squeeze(0)
            value = value.squeeze(0).item()

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

    # ── mode / persistence ──────────────────────────────────────────

    def set_train_mode(self, mode: bool):
        self.train_mode = mode
        self.model.train(mode)
        self.spy.train(mode)

    def save_model(self, path: str):
        torch.save({
            "model": self.model.state_dict(),
            "spy": self.spy.state_dict(),
        }, path)

    def load_model(self, path: str):
        checkpoint = torch.load(path, weights_only=False)
        if isinstance(checkpoint, dict) and "model" in checkpoint:
            self.model.load_state_dict(checkpoint["model"])
            if "spy" in checkpoint:
                self.spy.load_state_dict(checkpoint["spy"])
        else:
            # Backwards-compatible: bare state_dict for the actor-critic only
            self.model.load_state_dict(checkpoint)
