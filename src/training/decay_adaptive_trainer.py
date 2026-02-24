"""
Decay Adaptive Trainer — session-based training with EMA opponent statistics.

Extends AdaptiveTrainer to use DecayPokerSession instead of PokerSession.
The only change is the session object: DecayPokerSession provides EMA-weighted
opponent stats instead of uniform count-based stats.

All training logic (collect_episode, train loop, update_model, debug_episode)
is inherited from AdaptiveTrainer and works unchanged because DecayPokerSession
has the same interface as PokerSession.
"""

from typing import Dict

from src.agents.base import BaseAgent
from src.engine.decay_session import DecayPokerSession
from src.training.adaptive_trainer import AdaptiveTrainer


class DecayAdaptiveTrainer(AdaptiveTrainer):
    """Adaptive trainer using EMA-weighted opponent statistics.

    Overrides the session object to use DecayPokerSession (EMA stats)
    instead of PokerSession (uniform stats). Also supports an 'alpha'
    parameter for tuning the EMA decay rate.
    """

    def __init__(self, agent: BaseAgent, learning_rate: float = 1e-4,
                 hands_per_session: int = 30, alpha: float = 0.1):
        super().__init__(agent, learning_rate=learning_rate,
                         hands_per_session=hands_per_session)
        self.alpha = alpha
        self.session = DecayPokerSession(alpha=alpha)  # Replace PokerSession

    def update_params(self, params: Dict):
        """Updates trainer parameters, including EMA alpha."""
        super().update_params(params)
        if "alpha" in params:
            self.alpha = params["alpha"]
            self.session = DecayPokerSession(alpha=self.alpha)
            print(f"EMA alpha updated to: {self.alpha}")
