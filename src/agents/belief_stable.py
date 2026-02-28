"""
Belief Stable Agent for Leduc Hold'em.

Identical architecture to BeliefValueAgent (14-dim input, same network).
The ONLY difference is in the trainer: the TD target uses the OLD belief
(b_t) instead of the UPDATED belief (b_{t+1}).

This agent class is a thin wrapper that re-exports BeliefValueAgent's
behavior under a distinct type, so it can be registered and tracked
separately in the registry.
"""

from src.agents.belief_value import BeliefValueAgent


class BeliefStableAgent(BeliefValueAgent):
    """
    Bayesian Belief Agent with stable belief TD targets.

    Architecturally identical to BeliefValueAgent. The training change
    (using b_t instead of b_{t+1} in TD targets) is implemented in
    BeliefStableTrainer, not in the agent itself.
    """

    def __init__(self, model_path: str = None, temperature: float = 1.0):
        super().__init__(model_path=model_path, temperature=temperature)
