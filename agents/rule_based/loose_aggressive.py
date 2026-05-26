"""
LooseAggressiveAgent — Loose-Aggressive

Plays most hands and raises frequently. Uses aggression to win pots without
the best hand — the threat of bluffing is what makes strong hands pay off
more. Unlike the maniac, they do recognize when a bluff has failed: if they
bet J-high and get raised back, they cut their losses and fold rather than
throwing more chips into a lost pot.

Exploitable by: calling down with medium hands (they bluff a lot), but
risky — they do have strong hands sometimes and will punish thin calls.
"""

import random

from agents.base import BaseAgent
from engine.leduc_game import Action
from engine.observation import Observation

_PREFLOP_JACK_BLUFF = 0.40  # Raise J pre-flop 40% of the time
_FLOP_JACK_BLUFF    = 0.35  # Raise J-high on flop 35% (bluff attempt)
_FLOP_QUEEN_RERAISE = 0.45  # Re-raise Q-high on flop 45% (semi-bluff)


class LooseAggressiveAgent(BaseAgent):
    """
    Loose-Aggressive. Raises most hands; bluffs at meaningful frequencies.
    Folds J-high on the flop when a bluff has been called and raised back.
    """

    def __init__(self, seed: int = None):
        self._rng = random.Random(seed)

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
            return Action.RAISE if Action.RAISE in legal else Action.CALL
        if hand == 'Q':
            return Action.RAISE if Action.RAISE in legal else Action.CALL
        # Jack: bluff-raise 40%; otherwise call (loose — never fold pre-flop).
        if self._rng.random() < _PREFLOP_JACK_BLUFF and Action.RAISE in legal:
            return Action.RAISE
        return Action.CALL if Action.CALL in legal else Action.FOLD

    def _flop(self, hand, has_pair, facing_raise, legal):
        if has_pair:
            return Action.RAISE if Action.RAISE in legal else Action.CALL
        if hand == 'K':
            return Action.RAISE if Action.RAISE in legal else Action.CALL
        if hand == 'Q':
            if facing_raise:
                # Semi-bluff re-raise 45%; otherwise call — Q-high has some
                # showdown value so calling is not a disaster.
                if self._rng.random() < _FLOP_QUEEN_RERAISE and Action.RAISE in legal:
                    return Action.RAISE
                return Action.CALL if Action.CALL in legal else Action.FOLD
            return Action.RAISE if Action.RAISE in legal else Action.CALL
        # J-high on the flop.
        if facing_raise:
            # Bluff failed — facing a raise with nothing. Cut losses and fold.
            if Action.FOLD in legal:
                return Action.FOLD
            return Action.CALL if Action.CALL in legal else Action.FOLD
        # No raise yet — attempt a bluff 35% of the time; check/call otherwise.
        if self._rng.random() < _FLOP_JACK_BLUFF and Action.RAISE in legal:
            return Action.RAISE
        return Action.CALL if Action.CALL in legal else Action.FOLD
