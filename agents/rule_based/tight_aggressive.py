"""
TightAggressiveAgent — Tight-Aggressive

The "textbook good player." Selective about which hands to play,
but bets and raises aggressively when they do enter a pot. They don't
bluff — every raise signals genuine strength. Folds marginal hands to
pressure rather than calling down hoping to get lucky.

Exploitable by: calling their raises down with medium hands (they rarely
bluff), and by bluffing them off marginal holdings (they fold Q-high
and J-high to pressure).
"""

from agents.base import BaseAgent
from engine.leduc_game import Action
from engine.observation import Observation


class TightAggressiveAgent(BaseAgent):
    """
    Tight-Aggressive. Raises premium hands, folds marginal ones under pressure.
    No bluffing — aggression always represents real hand strength.
    """

    def select_action(self, obs: Observation) -> Action:
        hand = obs.player_hand
        board = obs.board
        legal = obs.legal_actions
        facing_raise = obs.raises_this_round > 0
        has_pair = board is not None and board == hand

        if obs.current_round == 0:
            return self._preflop(hand, facing_raise, legal)
        else:
            return self._flop(hand, has_pair, facing_raise, legal)

    def _preflop(self, hand, facing_raise, legal):
        if hand == 'K':
            # King is the best pre-flop hand — always raise.
            return Action.RAISE if Action.RAISE in legal else Action.CALL
        if hand == 'Q':
            # Queen: raise first-in; call (don't re-raise) facing a raise.
            # A re-raise with Q when K is out there is a bad spot.
            if facing_raise:
                return Action.CALL if Action.CALL in legal else Action.FOLD
            return Action.RAISE if Action.RAISE in legal else Action.CALL
        # Jack: fold to a raise (too weak); limp in otherwise.
        if facing_raise and Action.FOLD in legal:
            return Action.FOLD
        return Action.CALL if Action.CALL in legal else Action.FOLD

    def _flop(self, hand, has_pair, facing_raise, legal):
        # Pair = nuts — raise for value every time.
        if has_pair:
            return Action.RAISE if Action.RAISE in legal else Action.CALL
        # K-high: strong unimproved hand — raise for thin value.
        if hand == 'K':
            return Action.RAISE if Action.RAISE in legal else Action.CALL
        # Q-high: medium hand — call if uncontested; fold to a raise.
        if hand == 'Q':
            if facing_raise and Action.FOLD in legal:
                return Action.FOLD
            return Action.CALL if Action.CALL in legal else Action.FOLD
        # J-high: weakest hand — fold to any aggression; check/call if free.
        if facing_raise and Action.FOLD in legal:
            return Action.FOLD
        return Action.CALL if Action.CALL in legal else Action.FOLD
