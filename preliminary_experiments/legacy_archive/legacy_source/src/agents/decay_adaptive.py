"""
Decay-Weighted Adaptive Agent — adaptive value agent with EMA opponent stats.

Identical to AdaptiveValueAgent at inference time: same 19-dim encoding
(15 base + 4 stats), same ValueNetwork architecture, same action selection.

The difference is purely in training infrastructure: DecayAdaptiveTrainer uses
DecayPokerSession which provides EMA-weighted opponent statistics instead of
uniform count-based statistics. This means the 4 stat features (fold_rate,
raise_rate, fold_to_raise_rate, confidence) are computed with exponential
moving averages, giving more weight to recent opponent behavior.
"""

from .adaptive_value import AdaptiveValueAgent


class DecayAdaptiveAgent(AdaptiveValueAgent):
    """Adaptive value agent with EMA-weighted opponent statistics.

    Identical to AdaptiveValueAgent at inference time (same 19-dim encoding,
    same value network, same action selection). The difference is that the
    opponent stats features use exponential moving averages instead of
    uniform counts, giving more weight to recent opponent behavior.

    This is a thin subclass — everything is inherited from AdaptiveValueAgent.
    The behavioral difference comes from the training infrastructure
    (DecayAdaptiveTrainer + DecayPokerSession) which feeds EMA stats.
    """
    pass  # Everything inherited — the difference is in the trainer/session
