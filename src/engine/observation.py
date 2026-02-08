from dataclasses import dataclass, asdict
from typing import List, Optional, Any
from enum import IntEnum

@dataclass(frozen=True)
class Observation:
    """
    Structured game observation for Leduc Hold'em.
    Encapsulates state information from a specific player's perspective.
    """
    player_hand: str
    board: Optional[str]
    pot: List[int]
    current_player: int
    current_round: int
    legal_actions: List[Any]  # List of Action enums
    is_finished: bool

    def to_dict(self) -> dict:
        """Converts the observation to a dictionary for JSON serialization."""
        return asdict(self)
