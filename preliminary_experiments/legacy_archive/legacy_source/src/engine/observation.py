from dataclasses import dataclass, asdict
from typing import List, Optional, Any
from enum import IntEnum

@dataclass(frozen=True)
class Observation:
    """
    Structured game observation for Leduc Hold'em.
    Encapsulates state information from a specific player's perspective.

    The optional opponent_stats field carries cross-hand behavior statistics
    when playing within a PokerSession. It is None for single-hand play.
    """
    player_hand: str
    board: Optional[str]
    pot: List[int]
    current_player: int
    current_round: int
    legal_actions: List[Any]  # List of Action enums
    is_finished: bool
    raises_this_round: int = 0
    opponent_stats: Optional[Any] = None  # OpponentStats instance or None
    action_history: Optional[tuple] = None  # ((player, action_name), ...) for CFR infoset keys

    def to_dict(self) -> dict:
        """Converts the observation to a dictionary for JSON serialization."""
        d = asdict(self)
        if self.opponent_stats is not None and hasattr(self.opponent_stats, 'to_feature_vector'):
            d['opponent_stats'] = self.opponent_stats.to_feature_vector()
        return d
