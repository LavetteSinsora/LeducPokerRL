"""
Nash Value Agent -- value network trained on exact CFR equilibrium values.

This agent uses the same architecture and inference logic as ValueBasedAgent
(15-dim encoding, 64x64 hidden, 1-step lookahead). The ONLY difference is
HOW it was trained:
  - ValueBasedAgent: TD(0) self-play (noisy, non-stationary targets)
  - NashValueAgent: Supervised regression on exact Nash values from CFR

At inference time, behaviour is identical to ValueBasedAgent.
"""

from src.agents.value_based import ValueBasedAgent


class NashValueAgent(ValueBasedAgent):
    """Value agent whose network was trained on CFR Nash equilibrium values.

    No code changes from ValueBasedAgent -- the class exists so the registry
    can distinguish the two agents and so saved models carry semantic meaning.
    """

    def __init__(self, model_path: str = None, temperature: float = 1.0):
        super().__init__(model_path=model_path, temperature=temperature)
