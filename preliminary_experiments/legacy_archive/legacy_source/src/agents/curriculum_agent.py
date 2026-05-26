from .adaptive_value import AdaptiveValueAgent


class CurriculumAgent(AdaptiveValueAgent):
    """Adaptive agent trained with curriculum-based population training.

    Same architecture and encoding as AdaptiveValueAgent (19-dim input).
    The difference is in CurriculumTrainer which uses block scheduling,
    rehearsal buffers, and forgetting monitoring instead of random opponent
    rotation. This fixes the three main problems with PopAdaptiveAgent:
    conflicting gradients from diverse opponents, disrupted stat accumulation,
    and single-player training data.
    """
    pass
