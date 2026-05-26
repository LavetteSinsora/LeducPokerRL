"""
LoosePassiveAgent — Loose-Passive

The player who "can't fold." Justifies every call with sunk cost:
"I already put chips in, might as well see." They're passive because
they distrust their own hand strength — only raise when they're sure
they have the best hand (a pair). They never fold pre-flop (any two
cards can make a pair on the flop!), but even they recognize when
they're completely beaten on the flop and cut their losses.

Exploitable by: value betting any decent hand relentlessly — they will
call you down. Bluffing is mostly futile (they call too often).
"""

from agents.base import BaseAgent
from engine.leduc_game import Action
from engine.observation import Observation


class LoosePassiveAgent(BaseAgent):
    """
    Loose-Passive. Calls almost everything; folds only J-high on the flop
    facing a raise. Raises only with a made pair.
    """

    def select_action(self, obs: Observation) -> Action:
        hand = obs.player_hand
        board = obs.board
        legal = obs.legal_actions
        facing_raise = obs.raises_this_round > 0
        has_pair = board is not None and board == hand

        if obs.current_round == 0:
            return self._preflop(legal)
        else:
            return self._flop(hand, has_pair, facing_raise, legal)

    def _preflop(self, legal):
        # Never fold pre-flop — "any two cards can pair up!"
        return Action.CALL if Action.CALL in legal else legal[0]

    def _flop(self, hand, has_pair, facing_raise, legal):
        # Pair = nuts — raise it up even for a passive player.
        if has_pair:
            return Action.RAISE if Action.RAISE in legal else Action.CALL
        # K-high or Q-high: call everything.
        if hand in ('K', 'Q'):
            return Action.CALL if Action.CALL in legal else Action.FOLD
        # J-high = the absolute worst hand. Even a calling station folds
        # J-high to a raise — there's no draw, no pair, no hope.
        if facing_raise and Action.FOLD in legal:
            return Action.FOLD
        return Action.CALL if Action.CALL in legal else Action.FOLD
