"""
ManiacAgent — Pure Aggression

The maniac puts maximum pressure on every street. They raise every chance
they get regardless of cards — the goal is to bully opponents into folding.
They're not "bluffing" in a strategic sense; they just always escalate.

Even a maniac has one rational escape hatch: if they hold J-high (no pair,
worst possible hand), raises are capped for the round, and someone is still
applying pressure — they can't re-raise, and they have nothing. At that
point even the maniac recognizes there's no play left and folds.

Exploitable by: calling down with any decent hand. Their aggression is
unconditional, so you know a call will win at showdown more often than not.
"""

from agents.base import BaseAgent
from engine.leduc_game import Action
from engine.observation import Observation


class ManiacAgent(BaseAgent):
    """
    Always-Aggressive. Raises whenever possible; falls back to call.
    Only folds when holding J-high (no pair), raise is capped, and
    facing pressure — the one situation where there is literally no play.
    """

    def select_action(self, obs: Observation) -> Action:
        legal = obs.legal_actions
        hand = obs.player_hand
        board = obs.board
        facing_raise = obs.raises_this_round > 0
        has_pair = board is not None and board == hand
        raise_capped = Action.RAISE not in legal

        # Raise whenever possible — that's the maniac's default.
        if Action.RAISE in legal:
            return Action.RAISE

        # Raise is capped. Check for the one fold condition:
        # J-high (absolute worst hand), no pair, facing a raise, no way to re-raise.
        if not has_pair and hand == 'J' and facing_raise and Action.FOLD in legal:
            return Action.FOLD

        # All other situations: call.
        return Action.CALL if Action.CALL in legal else Action.FOLD
