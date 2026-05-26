"""
TightPassiveAgent — Tight-Passive

The "nit" archetype. Only invests chips with premium holdings, rarely raises,
folds to aggression with weak hands. Even a tight-passive player bets their
best hand (a pair) on the flop — the passive part means no bluffs or thin
value raises, not literally zero aggression ever.

Exploitable by: bluffing frequently (will fold marginal hands), and by
not raising back when they do call (you control the pot size against them).
"""

from agents.base import BaseAgent
from engine.leduc_game import Action
from engine.observation import Observation


class TightPassiveAgent(BaseAgent):
    """
    Tight-Passive. Calls with good hands, folds to pressure with weak ones.
    Raises only with a made pair (the nuts in Leduc) — never bluffs.
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
        # Pre-flop: no community card yet, so no pairs possible.
        # Calls with K or Q; folds J to a raise (too weak to defend),
        # calls J if no one has raised (cheap to see the flop).
        if hand == 'K':
            return Action.CALL if Action.CALL in legal else Action.FOLD
        if hand == 'Q':
            return Action.CALL if Action.CALL in legal else Action.FOLD
        # Jack
        if facing_raise and Action.FOLD in legal:
            return Action.FOLD
        return Action.CALL if Action.CALL in legal else Action.FOLD

    def _flop(self, hand, has_pair, facing_raise, legal):
        # Pair = the nuts in Leduc — even a tight-passive player bets their best hand.
        if has_pair:
            return Action.RAISE if Action.RAISE in legal else Action.CALL
        # K-high: decent hand, call down; not strong enough to raise without a pair.
        if hand == 'K':
            return Action.CALL if Action.CALL in legal else Action.FOLD
        # Q-high: medium hand, call if uncontested; fold to a raise.
        if hand == 'Q':
            if facing_raise and Action.FOLD in legal:
                return Action.FOLD
            return Action.CALL if Action.CALL in legal else Action.FOLD
        # J-high: worst hand; fold to any aggression.
        if facing_raise and Action.FOLD in legal:
            return Action.FOLD
        return Action.CALL if Action.CALL in legal else Action.FOLD
