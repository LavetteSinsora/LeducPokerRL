"""
ValueDeepAgent — DALI_modulation
==================================
Architecture: 15 → 64 → 64 → 64 → 1  (three hidden layers, depth ablation)

Identical to the canonical ValueBasedAgent except for an extra hidden layer.
Purpose: test whether the state_only / full_modulation gains are attributable
to extra network capacity rather than the frozen-base + residual structure.

If value_based_deep outperforms full_modulation (with same training budget),
the frozen-base design adds complexity without benefit; extra capacity alone
explains the improvement.
"""

import os
import sys
import torch
import torch.nn as nn
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, ROOT)

from agents.base import BaseAgent
from engine.leduc_game import Action, LeducGame
from engine.observation import Observation

GAME_DIMS  = 15
_CARD_MAP  = {'J': 0, 'Q': 1, 'K': 2}
_MAX_CHIPS = 13


class ValueDeepNetwork(nn.Module):
    """15 → 64 → 64 → 64 → 1 (3 hidden layers)."""

    def __init__(self, input_size: int = GAME_DIMS, hidden_size: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ValueDeepAgent(BaseAgent):
    """
    Deeper value-based agent. Same interface as ValueBasedAgent.
    Architecture: 15 → 64 → 64 → 64 → 1
    """

    def __init__(self, model_path: str = None, temperature: float = 1.0):
        self.temperature = temperature
        self.train_mode  = False

        self.net = ValueDeepNetwork(GAME_DIMS, hidden_size=64)
        if model_path:
            self.load_model(model_path)
        self.net.eval()

    def set_train_mode(self, mode: bool):
        self.train_mode = mode
        self.net.train(mode)

    def save_model(self, path: str):
        torch.save(self.net.state_dict(), path)

    def load_model(self, path: str):
        self.net.load_state_dict(
            torch.load(path, map_location="cpu", weights_only=True)
        )

    def _encode_game(self, obs: Observation, viewer_id: int) -> torch.Tensor:
        """15-dim game-state encoding (identical to FullModulationAgent)."""
        hand_idx = _CARD_MAP.get(obs.player_hand)
        hand_vec = torch.zeros(3)
        if hand_idx is not None:
            hand_vec[hand_idx] = 1.0

        board_idx = _CARD_MAP.get(obs.board, 3)
        board_vec = torch.zeros(4)
        board_vec[board_idx] = 1.0

        p0, p1 = obs.pot
        pot_rel = [p0, p1] if viewer_id == 0 else [p1, p0]
        pot_vec = torch.tensor(pot_rel, dtype=torch.float32) / _MAX_CHIPS

        feats = torch.tensor([
            1.0 if viewer_id == obs.current_player else 0.0,
            float(viewer_id),
            float(obs.current_round),
            1.0 if obs.is_finished else 0.0,
            1.0 if (obs.board is not None and obs.player_hand == obs.board) else 0.0,
            obs.raises_this_round / 2.0,
        ])
        return torch.cat([hand_vec, board_vec, pot_vec, feats])   # (15,)

    def _get_action_values(self, obs: Observation) -> list:
        cp = obs.current_player
        results = []
        for action in obs.legal_actions:
            post_obs, done = LeducGame.simulate_action(obs, action)
            if done and action == Action.FOLD:
                val = -float(obs.pot[cp])
            else:
                game_enc = self._encode_game(post_obs, cp).unsqueeze(0)
                with torch.no_grad():
                    val = self.net(game_enc).item()
            results.append({
                "action":   action,
                "value":    val,
                "game_enc": self._encode_game(post_obs, cp),
            })
        return results

    def select_action(self, obs: Observation) -> Action:
        results = self._get_action_values(obs)
        if not results:
            return Action.FOLD
        if self.train_mode:
            values = torch.tensor([r["value"] for r in results])
            probs  = torch.softmax(values / self.temperature, dim=0)
            idx    = torch.multinomial(probs, 1).item()
            return results[idx]["action"]
        return max(results, key=lambda r: r["value"])["action"]
