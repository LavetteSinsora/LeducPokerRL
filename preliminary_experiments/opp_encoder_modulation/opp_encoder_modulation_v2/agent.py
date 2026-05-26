"""Frozen base value agent with learned opponent encoder modulation."""

from dataclasses import replace
import math
from pathlib import Path

import torch
import torch.nn as nn

from agents.value_based.agent import ValueBasedAgent, ValueNetwork
from engine.leduc_game import Action, LeducGame
from engine.observation import Observation


DEFAULT_STATS = [0.5, 0.5, 0.5, 0.0]


class OpponentEncoder(nn.Module):
    """Compress macro opponent stats into a learned embedding."""

    def __init__(self, input_size: int = 4, hidden_size: int = 32, embedding_size: int = 8):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, embedding_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ResidualModulationNetwork(nn.Module):
    """Predict a bounded value correction from state plus opponent embedding."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 32,
        delta_scale: float = 1.0,
    ):
        super().__init__()
        self.delta_scale = delta_scale
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.delta_scale * torch.tanh(self.net(x))


class GateNetwork(nn.Module):
    """Learn how much of the residual to trust for a given embedding."""

    def __init__(self, input_size: int, hidden_size: int = 16, initial_gate: float = 0.4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid(),
        )
        bias = math.log(initial_gate / (1.0 - initial_gate))
        nn.init.constant_(self.net[2].bias, bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ActionPredictionHead(nn.Module):
    """Predict the opponent's next action from our information set plus embedding."""

    def __init__(self, input_size: int, hidden_size: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 3),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class OpponentEncoderModulationAgent(ValueBasedAgent):
    """Value agent with a frozen base net and learned opponent embedding."""

    STATS_SIZE = 4
    STATE_SIZE = 15

    def __init__(
        self,
        model_path: str = None,
        temperature: float = 1.0,
        base_model_path: str = None,
        embedding_size: int = 8,
        encoder_hidden_size: int = 32,
        modulation_hidden_size: int = 32,
        gate_hidden_size: int = 16,
        action_hidden_size: int = 32,
        delta_scale: float = 1.0,
        initial_gate: float = 0.4,
    ):
        self.input_size = self.STATE_SIZE
        self.temperature = temperature
        self.train_mode = False
        self.embedding_size = embedding_size

        self.model = ValueNetwork(self.input_size)
        self.encoder = OpponentEncoder(
            input_size=self.STATS_SIZE,
            hidden_size=encoder_hidden_size,
            embedding_size=embedding_size,
        )
        joint_size = self.STATE_SIZE + embedding_size
        self.mod_net = ResidualModulationNetwork(
            input_size=joint_size,
            hidden_size=modulation_hidden_size,
            delta_scale=delta_scale,
        )
        self.gate_net = GateNetwork(
            input_size=embedding_size,
            hidden_size=gate_hidden_size,
            initial_gate=initial_gate,
        )
        self.action_head = ActionPredictionHead(
            input_size=joint_size,
            hidden_size=action_hidden_size,
        )

        if base_model_path is None:
            base_model_path = self.default_base_model_path()
        if base_model_path and Path(base_model_path).exists():
            self.model.load_state_dict(torch.load(base_model_path))

        for param in self.model.parameters():
            param.requires_grad = False
        self.model.eval()

        if model_path:
            self.load_model(model_path)

    @staticmethod
    def default_base_model_path() -> str:
        root = Path(__file__).resolve().parents[2]
        return str(root / "agents" / "value_based" / "checkpoint.pt")

    def set_train_mode(self, mode: bool):
        self.train_mode = mode
        self.model.eval()
        self.encoder.train(mode)
        self.mod_net.train(mode)
        self.gate_net.train(mode)
        self.action_head.train(mode)

    def encode_observation(self, obs: Observation, viewer_id: int = None) -> torch.Tensor:
        return super().encode_observation(obs, viewer_id)

    def encode_macro_stats(self, obs: Observation) -> torch.Tensor:
        if obs.opponent_stats is not None and hasattr(obs.opponent_stats, "to_feature_vector"):
            return torch.tensor(obs.opponent_stats.to_feature_vector(), dtype=torch.float32)
        return torch.tensor(DEFAULT_STATS, dtype=torch.float32)

    def _ensure_batch(self, tensor: torch.Tensor) -> torch.Tensor:
        if tensor.dim() == 1:
            return tensor.unsqueeze(0)
        return tensor

    def encode_opponent(self, stats_vec: torch.Tensor) -> torch.Tensor:
        stats_batch = self._ensure_batch(stats_vec)
        return self.encoder(stats_batch)

    def predict_action_logits(
        self,
        state_encoding: torch.Tensor,
        stats_vec: torch.Tensor,
    ) -> torch.Tensor:
        state_batch = self._ensure_batch(state_encoding)
        embedding = self.encode_opponent(stats_vec)
        joint = torch.cat([state_batch, embedding], dim=-1)
        return self.action_head(joint)

    def predict_action_probs(
        self,
        state_encoding: torch.Tensor,
        stats_vec: torch.Tensor,
    ) -> torch.Tensor:
        logits = self.predict_action_logits(state_encoding, stats_vec)
        return torch.softmax(logits, dim=-1)

    def predict_value_from_encoded(
        self,
        state_encoding: torch.Tensor,
        stats_vec: torch.Tensor,
        return_parts: bool = False,
    ):
        state_batch = self._ensure_batch(state_encoding)
        embedding = self.encode_opponent(stats_vec)
        joint = torch.cat([state_batch, embedding], dim=-1)

        with torch.no_grad():
            base_value = self.model(state_batch)

        delta = self.mod_net(joint)
        gate = self.gate_net(embedding)
        value = base_value + gate * delta

        if return_parts:
            return {
                "value": value,
                "base_value": base_value,
                "delta": delta,
                "gate": gate,
                "embedding": embedding,
            }
        return value

    def explain_value(self, obs: Observation, viewer_id: int) -> dict:
        state = self.encode_observation(obs, viewer_id=viewer_id)
        stats = self.encode_macro_stats(obs)
        parts = self.predict_value_from_encoded(state, stats, return_parts=True)
        return {
            "value": parts["value"].item(),
            "base_value": parts["base_value"].item(),
            "delta": parts["delta"].item(),
            "gate": parts["gate"].item(),
            "embedding": parts["embedding"].squeeze(0).tolist(),
            "stats": stats.tolist(),
        }

    def _get_value(self, obs: Observation, viewer_id: int) -> float:
        state = self.encode_observation(obs, viewer_id=viewer_id)
        stats = self.encode_macro_stats(obs)
        with torch.no_grad():
            value = self.predict_value_from_encoded(state, stats)
        return value.item()

    def get_action_evaluations(self, obs: Observation) -> list:
        evaluations = []
        current_player = obs.current_player

        for action in obs.legal_actions:
            post_obs, done = LeducGame.simulate_action(obs, action)
            if obs.opponent_stats is not None:
                post_obs = replace(post_obs, opponent_stats=obs.opponent_stats)

            if done and action == Action.FOLD:
                value = -float(obs.pot[current_player])
                parts = {
                    "value": value,
                    "base_value": value,
                    "delta": 0.0,
                    "gate": 0.0,
                    "embedding": [],
                    "stats": self.encode_macro_stats(obs).tolist(),
                }
            else:
                parts = self.explain_value(post_obs, viewer_id=current_player)
                value = parts["value"]

            encoded = self.encode_observation(post_obs, viewer_id=current_player)
            evaluations.append(
                {
                    "action": action,
                    "value": value,
                    "is_terminal": done,
                    "encoded": encoded,
                    "base_value": parts["base_value"],
                    "delta": parts["delta"],
                    "gate": parts["gate"],
                }
            )
        return evaluations

    def save_model(self, path: str) -> None:
        torch.save(
            {
                "base": self.model.state_dict(),
                "encoder": self.encoder.state_dict(),
                "mod": self.mod_net.state_dict(),
                "gate": self.gate_net.state_dict(),
                "action_head": self.action_head.state_dict(),
                "embedding_size": self.embedding_size,
            },
            path,
        )

    def load_model(self, path: str) -> None:
        checkpoint = torch.load(path, weights_only=False)
        if isinstance(checkpoint, dict) and "encoder" in checkpoint:
            self.model.load_state_dict(checkpoint["base"])
            self.encoder.load_state_dict(checkpoint["encoder"])
            self.mod_net.load_state_dict(checkpoint["mod"])
            self.gate_net.load_state_dict(checkpoint["gate"])
            self.action_head.load_state_dict(checkpoint["action_head"])
            return

        # Backward compatibility: allow loading a plain value network as the base.
        self.model.load_state_dict(checkpoint)
