"""
Decay-Weighted Opponent Statistics — EMA-based opponent modeling.

Drop-in replacement for OpponentStats that uses exponential moving averages
instead of uniform count-based averaging. This gives more weight to recent
opponent actions, enabling faster adaptation when opponent strategy shifts.

Key difference from OpponentStats:
  - OpponentStats:      fold_rate = fold_count / total_actions  (uniform)
  - DecayOpponentStats: fold_rate = alpha * is_fold + (1-alpha) * old_fold_rate  (EMA)

Default alpha=0.1 means each new observation gets 10% weight, with the
remaining 90% coming from the exponentially-weighted history.
"""

from dataclasses import dataclass


@dataclass
class DecayOpponentStats:
    """Opponent stats with exponential moving average (EMA) weighting.

    Instead of fold_rate = fold_count / total_count (uniform average),
    uses EMA: fold_rate = alpha * is_fold + (1 - alpha) * old_fold_rate

    This gives more weight to recent observations, enabling faster
    adaptation when opponent strategy shifts.

    Interface-compatible with OpponentStats: same record_action(),
    record_hand_complete(), to_feature_vector(), reset(), and property
    accessors (fold_rate, raise_rate, fold_to_raise_rate).
    """

    alpha: float = 0.1  # EMA decay rate (higher = more weight on recent)

    # EMA smoothed rates — start at 0.5 (maximum entropy / uninformative)
    fold_rate_ema: float = 0.5
    raise_rate_ema: float = 0.5
    call_rate_ema: float = 0.5  # Not used in feature vector but tracked
    fold_to_raise_rate_ema: float = 0.5

    hands_observed: int = 0
    total_actions: int = 0

    def record_action(self, action_name: str, was_facing_raise: bool):
        """Record a single action taken by the tracked opponent.

        Updates EMA rates using one-hot encoding of the current action.
        """
        self.total_actions += 1

        # One-hot encoding of current action
        is_fold = 1.0 if action_name == "FOLD" else 0.0
        is_raise = 1.0 if action_name == "RAISE" else 0.0
        is_call = 1.0 if action_name == "CALL" else 0.0

        # EMA update: new_rate = alpha * observation + (1 - alpha) * old_rate
        self.fold_rate_ema = self.alpha * is_fold + (1 - self.alpha) * self.fold_rate_ema
        self.raise_rate_ema = self.alpha * is_raise + (1 - self.alpha) * self.raise_rate_ema
        self.call_rate_ema = self.alpha * is_call + (1 - self.alpha) * self.call_rate_ema

        # Conditional: fold when facing raise (only updated when relevant)
        if was_facing_raise:
            self.fold_to_raise_rate_ema = (
                self.alpha * is_fold + (1 - self.alpha) * self.fold_to_raise_rate_ema
            )

    def record_hand_complete(self):
        """Called when a hand finishes."""
        self.hands_observed += 1

    @property
    def fold_rate(self) -> float:
        return self.fold_rate_ema

    @property
    def raise_rate(self) -> float:
        return self.raise_rate_ema

    @property
    def fold_to_raise_rate(self) -> float:
        return self.fold_to_raise_rate_ema

    def to_feature_vector(self) -> list:
        """Returns [fold_rate, raise_rate, fold_to_raise_rate, confidence].

        Same 4-feature format as OpponentStats.to_feature_vector().
        The confidence signal (hands_observed / 50, clamped to 1.0) lets the
        network learn to ignore the other stats when few hands have been played.
        """
        return [
            self.fold_rate_ema,
            self.raise_rate_ema,
            self.fold_to_raise_rate_ema,
            min(self.hands_observed / 50.0, 1.0),
        ]

    def reset(self):
        """Clear all state for a new session."""
        self.fold_rate_ema = 0.5
        self.raise_rate_ema = 0.5
        self.call_rate_ema = 0.5
        self.fold_to_raise_rate_ema = 0.5
        self.hands_observed = 0
        self.total_actions = 0
