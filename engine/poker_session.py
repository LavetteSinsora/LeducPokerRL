"""
Poker Session — multi-hand game wrapper with cross-hand intelligence.

PokerSession wraps LeducGame the way a match wraps individual hands.
It has the same essential API (new_hand, step, get_observation) but
automatically tracks opponent behavior statistics and records per-hand
analytics for post-hoc analysis.

Usage:
    session = PokerSession()
    for _ in range(50):
        session.new_hand()
        while not session.is_finished:
            obs = session.get_observation()  # includes opponent stats
            action = agent.select_action(obs)
            session.step(action)

    # Analyze adaptation over the session
    analytics = session.get_analytics()

Developers choose between LeducGame (single-hand, no cross-hand state)
and PokerSession (multi-hand, with stats) based on whether they need
cross-hand intelligence.
"""

from dataclasses import dataclass, replace
from .leduc_game import LeducGame, Action


@dataclass
class OpponentStats:
    """
    Tracks aggregate opponent behavior across a session.

    All rates default to 0.5 (maximum entropy) when no data exists,
    signaling "no information" to the network.
    """
    # Raw action counts
    total_actions: int = 0
    fold_count: int = 0
    raise_count: int = 0
    call_count: int = 0

    # Conditional: how often opponent folds when facing a raise
    actions_facing_raise: int = 0
    folds_facing_raise: int = 0

    hands_observed: int = 0

    def record_action(self, action_name: str, was_facing_raise: bool):
        """Record a single action taken by the tracked opponent."""
        self.total_actions += 1
        if action_name == "FOLD":
            self.fold_count += 1
        elif action_name == "RAISE":
            self.raise_count += 1
        elif action_name == "CALL":
            self.call_count += 1

        if was_facing_raise:
            self.actions_facing_raise += 1
            if action_name == "FOLD":
                self.folds_facing_raise += 1

    def record_hand_complete(self):
        """Called when a hand finishes."""
        self.hands_observed += 1

    @property
    def fold_rate(self) -> float:
        return self.fold_count / self.total_actions if self.total_actions > 0 else 0.5

    @property
    def raise_rate(self) -> float:
        return self.raise_count / self.total_actions if self.total_actions > 0 else 0.5

    @property
    def fold_to_raise_rate(self) -> float:
        if self.actions_facing_raise > 0:
            return self.folds_facing_raise / self.actions_facing_raise
        return 0.5

    def to_feature_vector(self) -> list:
        """Returns [fold_rate, raise_rate, fold_to_raise_rate, confidence].

        The confidence signal (hands_observed / 50, clamped to 1.0) lets the
        network learn to ignore the other stats when few hands have been played.
        """
        return [
            self.fold_rate,
            self.raise_rate,
            self.fold_to_raise_rate,
            min(self.hands_observed / 50.0, 1.0),
        ]

    def reset(self):
        """Clear all counters for a new session."""
        self.total_actions = 0
        self.fold_count = 0
        self.raise_count = 0
        self.call_count = 0
        self.actions_facing_raise = 0
        self.folds_facing_raise = 0
        self.hands_observed = 0


class PokerSession:
    """
    Multi-hand poker session with automatic opponent stats tracking.

    Wraps LeducGame to provide the same game-playing API while accumulating
    cross-hand intelligence. Stats persist across hands within a session
    and reset when reset() is called.
    """

    def __init__(self):
        self.game = LeducGame()
        # stats[i] = what player i has observed about their opponent
        self.stats = [OpponentStats(), OpponentStats()]
        self.hands_played = 0
        self.hand_history = []

    # --- Game-like API ---

    def new_hand(self):
        """Start a new hand within the session. Stats persist."""
        self.game.reset()

    def step(self, action):
        """Execute action. Automatically tracks stats and records analytics."""
        player = self.game.current_player
        opponent = 1 - player

        # Detect if the acting player is responding to a raise
        was_facing_raise = self.game.pot[opponent] > self.game.pot[player]

        # The opponent learns about this player's action
        action_name = action.name if hasattr(action, 'name') else Action(action).name
        self.stats[opponent].record_action(action_name, was_facing_raise)

        result = self.game.step(action)

        if self.game.is_finished:
            for s in self.stats:
                s.record_hand_complete()
            self.hands_played += 1
            self._record_hand_analytics()

        return result

    def get_observation(self, viewer_id=None):
        """Returns observation enriched with this viewer's model of their opponent."""
        base_obs = self.game.get_observation(viewer_id)
        viewer = viewer_id if viewer_id is not None else self.game.current_player
        return replace(base_obs, opponent_stats=self.stats[viewer])

    # --- Session management ---

    def reset(self):
        """Full reset — stats, analytics, and game."""
        self.stats = [OpponentStats(), OpponentStats()]
        self.hands_played = 0
        self.hand_history = []

    # --- Properties delegated to game ---

    def get_reward(self):
        """Delegate to underlying game for API compatibility with LeducGame."""
        return self.game.get_reward()

    @property
    def is_finished(self):
        return self.game.is_finished

    @property
    def current_player(self):
        return self.game.current_player

    # --- Analytics ---

    def _record_hand_analytics(self):
        """Record a summary of the completed hand for post-hoc analysis."""
        self.hand_history.append({
            'hand_number': self.hands_played,
            'rewards': self.game.get_reward(),
            'winner': self.game.winner,
            'actions': list(self.game.history),
            'pot': list(self.game.pot),
            'stats_snapshot': [s.to_feature_vector() for s in self.stats],
        })

    def get_analytics(self):
        """Return full session analytics for visualization/analysis."""
        return {
            'hands_played': self.hands_played,
            'hand_history': self.hand_history,
            'final_stats': [s.to_feature_vector() for s in self.stats],
        }
