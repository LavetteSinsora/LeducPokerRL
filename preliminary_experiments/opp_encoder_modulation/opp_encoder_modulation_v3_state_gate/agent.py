"""Opponent encoder modulation with a state-conditioned gate."""

import torch

from experiments.opp_encoder_modulation_v2.agent import (
    GateNetwork,
    OpponentEncoderModulationAgent as V2Agent,
)


class OpponentEncoderModulationAgent(V2Agent):
    """Let the gate see state plus embedding, instead of embedding alone."""

    def __init__(
        self,
        model_path: str = None,
        gate_hidden_size: int = 16,
        initial_gate: float = 0.4,
        **kwargs,
    ):
        super().__init__(
            model_path=None,
            gate_hidden_size=gate_hidden_size,
            initial_gate=initial_gate,
            **kwargs,
        )
        joint_size = self.STATE_SIZE + self.embedding_size
        self.gate_net = GateNetwork(
            input_size=joint_size,
            hidden_size=gate_hidden_size,
            initial_gate=initial_gate,
        )
        if model_path:
            self.load_model(model_path)

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
        gate = self.gate_net(joint)
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
