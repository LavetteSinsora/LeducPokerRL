"""
Lightweight Leduc Hold'em engine for PIMC rollouts.

Identical game logic to engine.leduc_game.LeducGame but built for speed:
  - No deepcopy: use pack() / restore() with a plain Python tuple
  - No action history (rollouts don't need it)
  - No observation construction (rollouts don't need it)
  - __slots__ for faster attribute access

Usage pattern:
    rg = RolloutGame()
    rg.init_from_obs(obs, player_i=0, h_i='K', h_j='Q')
    state = rg.pack()                  # snapshot before k-rollout loop
    for _ in range(k):
        rg.restore(state)              # O(1) in-place reset
        ret = run_rollout(rg, ...)     # rg is mutated; restore before next
"""

from __future__ import annotations

import random

# Reuse only the Action enum from the engine — game logic is reimplemented here.
from engine.leduc_game import Action

# ── Constants ─────────────────────────────────────────────────────────────────

_CARD_TO_IDX: dict[str, int] = {'J': 0, 'Q': 1, 'K': 2}
_BET_AMOUNTS: tuple[int, int] = (2, 4)   # pre-flop, flop
_DECK: tuple[str, ...] = ('J', 'J', 'Q', 'Q', 'K', 'K')
_MAX_RAISES: int = 2


# ── RolloutGame ───────────────────────────────────────────────────────────────

class RolloutGame:
    """
    Minimal Leduc Hold'em engine for single PIMC rollouts.

    Game rules exactly match LeducGame:
      - Ante 1 per player; bet amounts [2, 4] per round; max 2 raises/round.
      - FOLD ends game (opponent wins).
      - CALL: if opponent has more in pot, match; if equal and acting second, round ends.
      - RAISE: put in opponent_pot + bet_amount; raises_this_round++.
      - Round transition: pre-flop ends → deal board card → post-flop.
      - Showdown: pair beats high card; same rank = tie.

    Reward semantics (same as LeducGame):
      winner == 0  →  [+pot[1], -pot[1]]   (P0 nets P1's chips)
      winner == 1  →  [-pot[0], +pot[0]]   (P1 nets P0's chips)
      tie          →  [0, 0]
    """

    __slots__ = (
        'h0', 'h1',
        'pot0', 'pot1',
        'board',
        'current_round',
        'current_player',
        'raises_this_round',
        'is_finished',
        'winner',
        '_round_betting_ended',
        '_pending_board',
    )

    def __init__(self) -> None:
        # Safe defaults; always overwritten by init_from_obs before use.
        self.h0 = self.h1 = 'J'
        self.pot0 = self.pot1 = 1
        self.board = None
        self.current_round = 0
        self.current_player = 0
        self.raises_this_round = 0
        self.is_finished = False
        self.winner = None
        self._round_betting_ended = False
        self._pending_board: str | None = None

    # ── Initialisation ────────────────────────────────────────────────────────

    def init_from_obs(self, obs, player_i: int, h_i: str, h_j: str) -> None:
        """
        Initialise game state from an Observation and explicit hand assignments.

        player_i  : the searching player (0 or 1)
        h_i       : player_i's private card
        h_j       : imagined opponent's private card

        Pre-samples the pending board card from remaining cards so that all k
        rollouts from the same state see the same board, matching LeducGame's
        deepcopy semantics (deck state preserved across copies).
        """
        if player_i == 0:
            self.h0, self.h1 = h_i, h_j
        else:
            self.h0, self.h1 = h_j, h_i

        self.pot0, self.pot1 = int(obs.pot[0]), int(obs.pot[1])
        self.board = obs.board
        self.current_round = obs.current_round
        self.current_player = obs.current_player
        self.raises_this_round = obs.raises_this_round
        self.is_finished = obs.is_finished
        self.winner = None
        self._round_betting_ended = False

        # Pre-sample board card for potential flop transition.
        if self.current_round == 0 and self.board is None:
            remaining = list(_DECK)
            remaining.remove(self.h0)
            remaining.remove(self.h1)
            self._pending_board = random.choice(remaining)
        else:
            self._pending_board = None

    # ── Pack / restore ────────────────────────────────────────────────────────

    def pack(self) -> tuple:
        """Snapshot full state as a plain tuple. O(1), no allocation beyond tuple."""
        return (
            self.h0, self.h1,
            self.pot0, self.pot1,
            self.board,
            self.current_round,
            self.current_player,
            self.raises_this_round,
            self.is_finished,
            self.winner,
            self._round_betting_ended,
            self._pending_board,
        )

    def restore(self, state: tuple) -> None:
        """Restore to a packed snapshot. O(1) field assignments, no allocation."""
        (
            self.h0, self.h1,
            self.pot0, self.pot1,
            self.board,
            self.current_round,
            self.current_player,
            self.raises_this_round,
            self.is_finished,
            self.winner,
            self._round_betting_ended,
            self._pending_board,
        ) = state

    # ── Game API ──────────────────────────────────────────────────────────────

    def get_legal_actions(self) -> list:
        actions = [Action.FOLD, Action.CALL]
        if self.raises_this_round < _MAX_RAISES:
            actions.append(Action.RAISE)
        return actions

    def step(self, action: Action):
        """
        Execute one action.

        Returns:
            reward_list  : [r_p0, r_p1]  — meaningful only when is_finished
            done         : bool
            act_eid      : int  — action event ID  (actor * 3 + int(action))
            deal_eid     : int | None — deal event ID (6 + card_idx) if flop dealt
            actor        : int — player who acted
        """
        actor = self.current_player
        act_eid = actor * 3 + int(action)   # == action_event_id(actor, int(action))
        deal_eid = None
        bet = _BET_AMOUNTS[self.current_round]

        if action == Action.FOLD:
            self.is_finished = True
            self.winner = 1 - actor
            return self.get_reward(), True, act_eid, deal_eid, actor

        if action == Action.CALL:
            other_pot = self.pot1 if actor == 0 else self.pot0
            my_pot   = self.pot0 if actor == 0 else self.pot1
            if other_pot > my_pot:
                # Call a raise: match the other player's pot
                if actor == 0:
                    self.pot0 = other_pot
                else:
                    self.pot1 = other_pot
                self._round_betting_ended = True
            else:
                # Check: only ends round if acting second (player 1 in original sense)
                if actor == 1:
                    self._round_betting_ended = True

        elif action == Action.RAISE:
            other_pot = self.pot1 if actor == 0 else self.pot0
            new_pot = other_pot + bet
            if actor == 0:
                self.pot0 = new_pot
            else:
                self.pot1 = new_pot
            self.raises_this_round += 1
            self._round_betting_ended = False

        self.current_player = 1 - actor

        if self._round_betting_ended:
            if self.current_round == 0:
                # Transition to flop
                self.board = self._pending_board
                self.current_round = 1
                self.current_player = 0
                self.raises_this_round = 0
                self._round_betting_ended = False
                deal_eid = 6 + _CARD_TO_IDX[self.board]  # == deal_event_id(board)
            else:
                self._showdown()

        return self.get_reward(), self.is_finished, act_eid, deal_eid, actor

    def get_reward(self) -> list:
        if not self.is_finished:
            return [0, 0]
        if self.winner == 0:
            return [self.pot1, -self.pot1]
        if self.winner == 1:
            return [-self.pot0, self.pot0]
        return [0, 0]  # tie

    # ── Internal ──────────────────────────────────────────────────────────────

    def _showdown(self) -> None:
        self.is_finished = True
        _V = {'J': 0, 'Q': 1, 'K': 2}
        board = self.board

        def score(h: str) -> int:
            return (10 + _V[h]) if h == board else _V[h]

        s0, s1 = score(self.h0), score(self.h1)
        if s0 > s1:
            self.winner = 0
        elif s1 > s0:
            self.winner = 1
        else:
            self.winner = -1   # tie
