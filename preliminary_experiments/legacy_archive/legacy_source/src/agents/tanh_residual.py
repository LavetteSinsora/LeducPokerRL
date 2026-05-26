"""
Tanh-Bounded Residual Agent — frozen base + architecturally bounded delta.

Instead of a gate to constrain corrections, uses tanh to hard-bound delta output
to [-max_correction, +max_correction]. This avoids the gate's gradient attenuation
while providing a strict bound on how much the correction can deviate from the base.

Architecture:
    V(s, opp) = V_base(s) + tanh(raw_delta(s, opp_stats)) * max_correction
"""

import torch
import torch.nn as nn
from dataclasses import replace
from src.engine.leduc_game import Action, LeducGame
from src.engine.observation import Observation
from .value_based import ValueBasedAgent, ValueNetwork


class BoundedDeltaNetwork(nn.Module):
    """Delta network with tanh output bounding."""

    def __init__(self, input_size=19, hidden_size=32, max_correction=0.5):
        super().__init__()
        self.max_correction = max_correction
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, x):
        return torch.tanh(self.net(x)) * self.max_correction


class TanhResidualAgent(ValueBasedAgent):
    """Frozen base + tanh-bounded residual correction, no gate."""

    STATS_SIZE = 4

    def __init__(self, model_path=None, temperature=1.0, base_model_path=None,
                 max_correction=0.5):
        self.input_size = 15
        self.temperature = temperature
        self.train_mode = False

        self.model = ValueNetwork(self.input_size)
        self.delta_net = BoundedDeltaNetwork(15 + self.STATS_SIZE,
                                              max_correction=max_correction)

        if base_model_path:
            self.model.load_state_dict(torch.load(base_model_path))

        for p in self.model.parameters():
            p.requires_grad = False

        if model_path:
            self.load_model(model_path)

        self.model.eval()

    def encode_observation(self, obs: Observation, viewer_id: int = None) -> torch.Tensor:
        return super().encode_observation(obs, viewer_id)

    def _encode_stats(self, obs: Observation) -> torch.Tensor:
        if obs.opponent_stats is not None and hasattr(obs.opponent_stats, 'to_feature_vector'):
            return torch.tensor(obs.opponent_stats.to_feature_vector(), dtype=torch.float32)
        return torch.tensor([0.5, 0.5, 0.5, 0.0], dtype=torch.float32)

    def _get_value(self, obs: Observation, viewer_id: int) -> float:
        base_enc = self.encode_observation(obs, viewer_id=viewer_id)
        stats_vec = self._encode_stats(obs)

        with torch.no_grad():
            v_base = self.model(base_enc)
            mod_input = torch.cat([base_enc.squeeze(0), stats_vec]).unsqueeze(0)
            delta = self.delta_net(mod_input)
            return (v_base + delta).item()

    def get_action_evaluations(self, obs: Observation) -> list:
        evaluations = []
        current_p = obs.current_player

        for action in obs.legal_actions:
            post_obs, done = LeducGame.simulate_action(obs, action)
            if obs.opponent_stats is not None:
                post_obs = replace(post_obs, opponent_stats=obs.opponent_stats)

            if done and action == Action.FOLD:
                val = -float(obs.pot[current_p])
            else:
                val = self._get_value(post_obs, viewer_id=current_p)

            encoded = self.encode_observation(post_obs, viewer_id=current_p)
            evaluations.append({
                "action": action, "value": val,
                "is_terminal": done, "encoded": encoded,
            })
        return evaluations

    def set_train_mode(self, mode: bool):
        self.train_mode = mode
        self.model.eval()
        self.delta_net.train(mode)

    def save_model(self, path: str) -> None:
        torch.save({
            "base": self.model.state_dict(),
            "delta": self.delta_net.state_dict(),
        }, path)

    def load_model(self, path: str) -> None:
        data = torch.load(path)
        if isinstance(data, dict) and "base" in data:
            self.model.load_state_dict(data["base"])
            self.delta_net.load_state_dict(data["delta"])
        else:
            self.model.load_state_dict(data)
