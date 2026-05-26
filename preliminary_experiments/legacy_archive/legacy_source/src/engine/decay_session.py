"""
Decay Poker Session — PokerSession variant with EMA-based opponent statistics.

Drop-in replacement for PokerSession that uses DecayOpponentStats instead of
OpponentStats. All game-playing methods (new_hand, step, get_observation) are
inherited from PokerSession and work because DecayOpponentStats has the same
interface as OpponentStats.
"""

from .leduc_game import LeducGame
from .poker_session import PokerSession
from .decay_stats import DecayOpponentStats


class DecayPokerSession(PokerSession):
    """PokerSession that uses EMA-based opponent statistics.

    The only difference from PokerSession is the stats objects:
    DecayOpponentStats (EMA) instead of OpponentStats (uniform counts).
    """

    def __init__(self, alpha: float = 0.1):
        self.game = LeducGame()
        self.stats = [DecayOpponentStats(alpha=alpha), DecayOpponentStats(alpha=alpha)]
        self.hands_played = 0
        self.hand_history = []
        self.alpha = alpha

    def reset(self):
        """Full reset — stats, analytics, and game."""
        self.stats = [DecayOpponentStats(alpha=self.alpha), DecayOpponentStats(alpha=self.alpha)]
        self.hands_played = 0
        self.hand_history = []
