from .adaptive_value import AdaptiveValueAgent


class ExtendedAdaptiveAgent(AdaptiveValueAgent):
    """Adaptive value agent trained with extended budget (3-5x longer).

    Null hypothesis control for Round 3. If more training is all that's needed,
    all algorithmic changes are unnecessary.
    """
    pass
