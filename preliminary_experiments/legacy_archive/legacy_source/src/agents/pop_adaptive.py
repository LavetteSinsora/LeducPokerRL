from .adaptive_value import AdaptiveValueAgent


class PopAdaptiveAgent(AdaptiveValueAgent):
    """Adaptive value agent trained against a diverse opponent population.

    Same architecture and encoding as AdaptiveValueAgent (19-dim input).
    The difference is in PopAdaptiveTrainer which rotates through a pool
    of diverse opponents during training, giving the opponent_stats features
    genuine diversity to learn from.
    """
    pass
