"""Experiment package for opponent-encoder modulation v2."""

from experiments.opp_encoder_modulation_v2.agent import OpponentEncoderModulationAgent
from experiments.opp_encoder_modulation_v2.trainer import (
    OpponentEncoderModulationTrainer,
)

__all__ = [
    "OpponentEncoderModulationAgent",
    "OpponentEncoderModulationTrainer",
]
