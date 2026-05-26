"""Experiment package for opponent-encoder modulation v1."""

from experiments.opp_encoder_modulation_v1.agent import OpponentEncoderModulationAgent
from experiments.opp_encoder_modulation_v1.trainer import (
    OpponentEncoderModulationTrainer,
)

__all__ = [
    "OpponentEncoderModulationAgent",
    "OpponentEncoderModulationTrainer",
]
