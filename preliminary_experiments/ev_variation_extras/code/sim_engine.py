"""
FixedStateSimulator — custom Monte Carlo simulation engine for EV_variation_analysis.

Runs rollouts from a fully-specified game state where both player hands are
fixed. Board card randomness in pre-flop states is eliminated by proportionally
distributing rollouts across all possible board cards (weighted by card-removal
probabilities).

The observation fed to each agent is obtained via game.get_observation(viewer_id=acting),
which is identical to what agents receive during normal evaluation. No information
structure is altered.
"""

from collections import Counter

import numpy as np

from engine.leduc_game import LeducGame
from agents.base import BaseAgent


class FixedStateSimulator:
    """
    Monte Carlo rollout engine for a fully-specified game state.

    Both player hands (hand0 for player 0, hand1 for player 1) are fixed.
    The value agent always occupies the current_player seat at the injected
    state; the opponent agent occupies the other seat.

    Args:
        hand0: Card rank of player 0 ('J', 'Q', or 'K').
        hand1: Card rank of player 1 ('J', 'Q', or 'K').
        pot:   [player_0_chips, player_1_chips] at the time of injection.
        current_player: Which player acts first from this state (0 or 1).
        rnd:   Betting round — 0 = pre-flop, 1 = flop.
        raises: Number of raises already taken in the current round.
        board:  Community card rank for flop states; None for pre-flop states.
    """

    FULL_DECK = ['J', 'J', 'Q', 'Q', 'K', 'K']

    def __init__(
        self,
        hand0: str,
        hand1: str,
        pot: list,
        current_player: int,
        rnd: int,
        raises: int,
        board: str | None = None,
    ):
        self.hand0 = hand0
        self.hand1 = hand1
        self.pot = pot
        self.cp = current_player
        self.rnd = rnd
        self.raises = raises
        self.board = board  # None for pre-flop states

    # ------------------------------------------------------------------
    # Board distribution (pre-flop only)
    # ------------------------------------------------------------------

    def _board_distribution(self) -> dict:
        """
        Computes {board_card_rank: probability} for pre-flop states.

        Removes hand0 and hand1 from the 6-card Leduc deck (card-removal
        assumption); the 4 remaining cards form the pool from which the
        board is uniformly drawn.

        Examples:
          hand0=J, hand1=J  → remaining=[Q,Q,K,K] → {Q:0.5, K:0.5}
          hand0=J, hand1=Q  → remaining=[J,Q,K,K] → {J:0.25, Q:0.25, K:0.5}
        """
        remaining = list(self.FULL_DECK)
        remaining.remove(self.hand0)   # removes first occurrence of hand0
        remaining.remove(self.hand1)   # removes first occurrence of hand1
        counts = Counter(remaining)    # always sums to 4
        total = len(remaining)
        return {card: count / total for card, count in counts.items()}

    # ------------------------------------------------------------------
    # State injection
    # ------------------------------------------------------------------

    def _inject(self, game: LeducGame, board_for_deck: str | None = None) -> None:
        """
        Writes the fixed state directly into a LeducGame instance.

        Bypasses reset() so that both player hands can be set simultaneously.
        board_for_deck, when provided, is pre-loaded into game.deck so that
        _transition_to_flop() deterministically pops it as the community card.
        """
        game.player_hands = [self.hand0, self.hand1]
        game.board = self.board
        game.pot = list(self.pot)
        game.current_player = self.cp
        game.current_round = self.rnd
        game.raises_this_round = self.raises
        game.is_finished = False
        game.winner = None
        game.history = []
        game.round_betting_ended = False
        game.last_action_was_raise = (self.raises > 0)
        # Pre-seed deck: _transition_to_flop does self.board = self.deck.pop()
        game.deck = [board_for_deck] if board_for_deck is not None else []

    # ------------------------------------------------------------------
    # Single game playthrough
    # ------------------------------------------------------------------

    def _play_one(
        self,
        game: LeducGame,
        value_agent: BaseAgent,
        opponent: BaseAgent,
        value_player: int,
    ) -> float:
        """
        Play one game to terminal from the already-injected state.

        Observations are obtained via game.get_observation(viewer_id=acting),
        which is identical to the normal evaluation pipeline. Each agent
        receives only its own private information.

        Returns the reward for value_player at game end.
        """
        while not game.is_finished:
            acting = game.current_player
            obs = game.get_observation(viewer_id=acting)
            if acting == value_player:
                action = value_agent.select_action(obs)
            else:
                action = opponent.select_action(obs)
            game.step(action)
        return float(game.get_reward()[value_player])

    # ------------------------------------------------------------------
    # Main rollout API
    # ------------------------------------------------------------------

    def rollout(
        self,
        value_agent: BaseAgent,
        opponent: BaseAgent,
        n_rollouts: int = 200,
    ) -> tuple:
        """
        Run n_rollouts Monte Carlo simulations from this fixed state.

        Pre-flop states: rollouts are distributed proportionally across all
        possible board cards, weighted by their card-removal probabilities.
        This eliminates board-card variance from the EV estimate so that
        differences across opponents reflect strategy only.

        Flop states: the board is already fixed; n_rollouts are run directly.

        Args:
            value_agent: The agent whose EV is being measured. Always plays
                         from the current_player seat.
            opponent:    The opponent agent in the other seat.
            n_rollouts:  Target number of rollouts (actual may differ slightly
                         for pre-flop states due to rounding).

        Returns:
            (ev_mean, ev_std, n_actual)
              ev_mean:  Average reward for value_agent across all rollouts.
              ev_std:   Sample standard deviation (ddof=1).
              n_actual: Actual number of rollouts run.
        """
        value_player = self.cp
        rewards = []

        if self.rnd == 0:
            # Pre-flop: proportional board card sampling
            board_dist = self._board_distribution()
            for board_card, prob in board_dist.items():
                n_sub = max(1, round(n_rollouts * prob))
                for _ in range(n_sub):
                    game = LeducGame()
                    self._inject(game, board_for_deck=board_card)
                    r = self._play_one(game, value_agent, opponent, value_player)
                    rewards.append(r)
        else:
            # Flop: board is fixed
            for _ in range(n_rollouts):
                game = LeducGame()
                self._inject(game)
                r = self._play_one(game, value_agent, opponent, value_player)
                rewards.append(r)

        arr = np.array(rewards, dtype=np.float64)
        n = len(arr)
        ev_mean = float(arr.mean())
        ev_std = float(arr.std(ddof=1)) if n > 1 else 0.0
        return ev_mean, ev_std, n
