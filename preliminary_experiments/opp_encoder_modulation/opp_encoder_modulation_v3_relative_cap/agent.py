"""Opponent encoder modulation with a direct cap on effective residual size."""

from pathlib import Path

import torch

from experiments.opp_encoder_modulation_v2.agent import OpponentEncoderModulationAgent as V2Agent


class OpponentEncoderModulationAgent(V2Agent):
    """Keep v2 intact, but clamp the effective residual relative to base value size."""

    def __init__(
        self,
        model_path: str = None,
        relative_effective_cap: float = 0.1,
        cap_bias: float = 0.5,
        **kwargs,
    ):
        self.relative_effective_cap = relative_effective_cap
        self.cap_bias = cap_bias
        super().__init__(model_path=model_path, **kwargs)

    @staticmethod
    def default_base_model_path() -> str:
        root = Path(__file__).resolve().parents[2]
        return str(root / "agents" / "value_based" / "checkpoint.pt")

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
        uncapped_effective_delta = gate * delta
        cap = self.relative_effective_cap * (base_value.abs().detach() + self.cap_bias)
        effective_delta = torch.clamp(uncapped_effective_delta, -cap, cap)
        value = base_value + effective_delta

        if return_parts:
            return {
                "value": value,
                "base_value": base_value,
                "delta": delta,
                "gate": gate,
                "embedding": embedding,
                "effective_delta": effective_delta,
                "effective_cap": cap,
                "uncapped_effective_delta": uncapped_effective_delta,
            }
        return value
