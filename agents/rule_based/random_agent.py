"""
RandomAgent — Uniform Random

Selects uniformly at random from the legal actions at each decision point.
Useful as a baseline and sanity check; represents zero strategic knowledge.
"""

import random

from agents.base import BaseAgent
from engine.leduc_game import Action
from engine.observation import Observation


class RandomAgent(BaseAgent):
    """
    Baseline agent. Chooses a legal action uniformly at random each step.
    """

    def __init__(self, seed: int = None):
        self._rng = random.Random(seed)

    def select_action(self, obs: Observation) -> Action:
        return self._rng.choice(obs.legal_actions)
