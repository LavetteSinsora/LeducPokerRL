"""
Residual Value Agent — frozen base + unbounded delta, no gate.

The modulated_value architecture used gate(stats) * delta(state, stats), but the
gate learned a near-constant ~0.4, creating gradient starvation: delta gets only
40% of already-small residual gradients. Removing the gate gives delta the full
gradient signal while L2 regularization (weight_decay in Adam) keeps corrections
bounded.

Architecture:
    V(s, opp) = V_base(s) + Delta(s, opp_stats)

where:
    V_base  = frozen 15-dim value network (pretrained)
    Delta   = trainable modulation network (19-dim: game state + opponent stats)
"""

import torch
from dataclasses import replace
from src.engine.leduc_game import Action, LeducGame
from src.engine.observation import Observation
from .value_based import ValueBasedAgent, ValueNetwork
from .modulated_value import ModulationNetwork


class ResidualValueAgent(ValueBasedAgent):
    """
    Value agent with direct residual correction on a frozen base.

    Unlike ModulatedValueAgent, there is no gate — delta gets full gradient
    signal. Corrections are kept bounded via L2 regularization in training.
    """

    STATS_SIZE = 4

    def __init__(self, model_path=None, temperature=1.0, base_model_path=None):
        self.input_size = 15
        self.temperature = temperature
        self.train_mode = False

        self.model = ValueNetwork(self.input_size)  # frozen base
        self.delta_net = ModulationNetwork(15 + self.STATS_SIZE)

        if base_model_path:
            self.model.load_state_dict(torch.load(base_model_path))

        # Freeze base network
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
                "action": action,
                "value": val,
                "is_terminal": done,
                "encoded": encoded,
            })
        return evaluations

    def set_train_mode(self, mode: bool):
        self.train_mode = mode
        self.model.eval()  # base always frozen
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
