"""
Leduc Hold'em CFR+ Solver.

Implements Counterfactual Regret Minimization with:
- CFR+ regret flooring (negative regrets clipped to 0)
- Linear averaging (later iterations weighted more heavily)
- Full game tree traversal (no sampling — feasible for Leduc's small tree)
- Exploitability computation via best-response traversal

The solver is decoupled from the agent and training infrastructure.
It operates on a StrategyStore and pure game-state parameters (no LeducGame objects).
"""

from typing import Dict, List, Tuple

import numpy as np

from src.engine.leduc_game import Action
from src.cfr.strategy import StrategyStore, NUM_ACTIONS

CARDS = ["J", "Q", "K"]
CARD_VALUES = {"J": 0, "Q": 1, "K": 2}
BET_AMOUNTS = [2, 4]   # Pre-flop, Flop
MAX_RAISES = 2


def _generate_deals() -> List[Tuple[str, str, str, float]]:
    """Generate all (p0_hand, p1_hand, board_card, probability) tuples.

    Groups equivalent deals by card rank (not identity) since the two
    Jacks, two Queens, and two Kings are indistinguishable.
    """
    deals = []
    for c0 in CARDS:
        for c1 in CARDS:
            deal_weight = 2.0 / 30.0 if c0 == c1 else 4.0 / 30.0
            remaining = {}
            for c in CARDS:
                count = 2 - (c == c0) - (c == c1)
                if count > 0:
                    remaining[c] = count
            total_remaining = sum(remaining.values())
            for board_card, board_count in remaining.items():
                prob = deal_weight * board_count / total_remaining
                deals.append((c0, c1, board_card, prob))
    return deals


class LeducCFRSolver:
    """CFR+ solver for Leduc Hold'em.

    Each iteration:
    1. Snapshot current strategies (from regret sums)
    2. Traverse full game tree, accumulating regret deltas in a buffer
    3. Apply buffered deltas to regret sums, then floor to 0 (CFR+)

    Usage:
        store = TabularStrategyStore()
        solver = LeducCFRSolver(store)
        for i in range(1, 10001):
            solver.run_iteration(i)
        print(solver.compute_exploitability())
    """

    def __init__(self, strategy_store: StrategyStore):
        self.store = strategy_store
        self.deals = _generate_deals()
        # Buffers populated per iteration
        self._regret_buffer: Dict[str, np.ndarray] = {}
        self._strategy_cache: Dict[str, np.ndarray] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run_iteration(self, iteration: int) -> float:
        """Run one full CFR+ iteration. Returns expected game value for P0."""
        self._regret_buffer.clear()
        self._strategy_cache.clear()

        total_value = 0.0
        for p0_hand, p1_hand, board_card, chance_prob in self.deals:
            value = self._cfr(
                p0_hand, p1_hand, board_card, chance_prob,
                "", "", 0, 0, 1, 1, 0, 1.0, 1.0, iteration,
            )
            total_value += chance_prob * value

        # Apply buffered regret deltas and floor (CFR+)
        for key, delta in self._regret_buffer.items():
            info = self.store.get_info_set(key)
            info.regret_sum += delta
            np.maximum(info.regret_sum, 0.0, out=info.regret_sum)

        return total_value

    def compute_exploitability(self) -> float:
        """Exploitability = how much each player can gain by best-responding.

        Uses infoset-aware best response: the BR player averages over hidden
        opponent hands before choosing actions, so exploitability converges
        toward 0 as the average strategy approaches Nash equilibrium.
        """
        br_0 = self._compute_br_value(0)  # P0's value when P0 best-responds
        br_1 = self._compute_br_value(1)  # P0's value when P1 best-responds
        return br_0 - br_1

    def _compute_br_value(self, br_player: int) -> float:
        """Compute P0's expected value when br_player plays best response."""
        total = 0.0
        for br_hand in CARDS:
            hand_prob = 2.0 / 6.0
            opp_probs = {h: (2 - (h == br_hand)) / 5.0 for h in CARDS}
            val = self._br_node(
                br_hand, opp_probs, "",
                "", "", 0, 0, 1, 1, 0, br_player,
            )
            total += hand_prob * val
        return total

    # ------------------------------------------------------------------
    # Core CFR traversal
    # ------------------------------------------------------------------

    def _cfr(
        self,
        p0_hand: str, p1_hand: str, board: str, chance_prob: float,
        preflop: str, flop: str,
        rnd: int, player: int,
        pot0: int, pot1: int, raises: int,
        r0: float, r1: float,
        iteration: int,
    ) -> float:
        """Recursive CFR+ traversal. Returns expected value for player 0.

        r0, r1: reach probabilities for player 0 and player 1 respectively.
        These track how likely each player is to reach this node under their
        current strategy, and are used to weight regret and strategy updates.
        """
        hand = p0_hand if player == 0 else p1_hand
        key = _make_key(hand, board, preflop, flop, rnd)
        legal = _legal_actions(raises)

        # Use cached strategy (consistent within one iteration)
        if key not in self._strategy_cache:
            self._strategy_cache[key] = self.store.get_strategy(key, legal)
        strategy = self._strategy_cache[key]

        action_vals = np.zeros(NUM_ACTIONS, dtype=np.float64)
        node_val = 0.0

        for action in legal:
            a = action.value
            new_r0 = r0 * (strategy[a] if player == 0 else 1.0)
            new_r1 = r1 * (strategy[a] if player == 1 else 1.0)
            v = self._apply(
                action, p0_hand, p1_hand, board, chance_prob,
                preflop, flop, rnd, player,
                pot0, pot1, raises, new_r0, new_r1, iteration,
            )
            action_vals[a] = v
            node_val += strategy[a] * v

        # Buffer regret deltas (weighted by chance and opponent reach)
        opp_reach = r1 if player == 0 else r0
        sign = 1.0 if player == 0 else -1.0
        if key not in self._regret_buffer:
            self._regret_buffer[key] = np.zeros(NUM_ACTIONS, dtype=np.float64)
        for action in legal:
            a = action.value
            self._regret_buffer[key][a] += chance_prob * opp_reach * sign * (action_vals[a] - node_val)

        # Strategy sum: weighted by iteration (linear CFR+ averaging) and own reach
        info = self.store.get_info_set(key)
        my_reach = r0 if player == 0 else r1
        for action in legal:
            a = action.value
            info.strategy_sum[a] += iteration * my_reach * strategy[a]

        return node_val

    def _apply(
        self,
        action: Action,
        p0_hand: str, p1_hand: str, board: str, chance_prob: float,
        preflop: str, flop: str,
        rnd: int, player: int,
        pot0: int, pot1: int, raises: int,
        r0: float, r1: float,
        iteration: int,
    ) -> float:
        """Apply action, return terminal value or recurse."""
        code = "fcr"[action.value]
        pf = preflop + code if rnd == 0 else preflop
        fl = flop + code if rnd == 1 else flop

        # Fold
        if action == Action.FOLD:
            return -pot0 if player == 0 else pot1

        other_pot = pot1 if player == 0 else pot0
        my_pot = pot0 if player == 0 else pot1
        new_pot0, new_pot1 = pot0, pot1

        # Raise
        if action == Action.RAISE:
            new_my = other_pot + BET_AMOUNTS[rnd]
            if player == 0:
                new_pot0 = new_my
            else:
                new_pot1 = new_my
            return self._cfr(
                p0_hand, p1_hand, board, chance_prob,
                pf, fl, rnd, 1 - player,
                new_pot0, new_pot1, raises + 1,
                r0, r1, iteration,
            )

        # Call / Check
        round_ended = False
        if other_pot > my_pot:
            if player == 0:
                new_pot0 = other_pot
            else:
                new_pot1 = other_pot
            round_ended = True
        elif player == 1:
            round_ended = True

        if not round_ended:
            return self._cfr(
                p0_hand, p1_hand, board, chance_prob,
                pf, fl, rnd, 1 - player,
                new_pot0, new_pot1, raises,
                r0, r1, iteration,
            )

        # Round ended
        if rnd == 0:
            return self._cfr(
                p0_hand, p1_hand, board, chance_prob,
                pf, "", 1, 0,
                new_pot0, new_pot1, 0,
                r0, r1, iteration,
            )
        return _showdown(p0_hand, p1_hand, board, new_pot0, new_pot1)

    # ------------------------------------------------------------------
    # Infoset-aware best-response traversal (for exploitability)
    # ------------------------------------------------------------------

    def _br_node(
        self,
        br_hand: str, opp_probs: Dict[str, float], board: str,
        preflop: str, flop: str,
        rnd: int, player: int,
        pot0: int, pot1: int, raises: int,
        br_player: int,
    ) -> float:
        """Infoset-aware BR traversal. Returns P0's expected value.

        br_hand:   the BR player's card
        opp_probs: probability distribution over opponent's card {rank: prob}
        board:     "" during pre-flop, a specific card during flop
        """
        legal = _legal_actions(raises)

        if player == br_player:
            # BR player picks best action (averaged over hidden opponent info)
            action_vals = {}
            for action in legal:
                action_vals[action.value] = self._br_after_action(
                    action, br_hand, opp_probs, board,
                    preflop, flop, rnd, player,
                    pot0, pot1, raises, br_player,
                )
            return max(action_vals.values()) if br_player == 0 else min(action_vals.values())

        # Opponent plays average strategy. Each possible opponent hand h plays
        # according to its own infoset's average strategy. After the opponent
        # acts, we Bayesian-update the distribution: hands that would play
        # this action with higher probability become more likely.
        total_val = 0.0
        for action in legal:
            a = action.value
            action_weight = 0.0
            new_opp: Dict[str, float] = {}
            for h, p in opp_probs.items():
                if p <= 0:
                    continue
                key = _make_key(h, board, preflop, flop, rnd)
                avg = self.store.get_average_strategy(key, legal)
                w = p * avg[a]
                if w > 0:
                    new_opp[h] = w
                    action_weight += w
            if action_weight <= 0:
                continue
            for h in new_opp:
                new_opp[h] /= action_weight
            v = self._br_after_action(
                action, br_hand, new_opp, board,
                preflop, flop, rnd, player,
                pot0, pot1, raises, br_player,
            )
            total_val += action_weight * v
        return total_val

    def _br_after_action(
        self,
        action: Action,
        br_hand: str, opp_probs: Dict[str, float], board: str,
        preflop: str, flop: str,
        rnd: int, player: int,
        pot0: int, pot1: int, raises: int,
        br_player: int,
    ) -> float:
        """Apply action in BR traversal, handling chance nodes and terminals."""
        code = "fcr"[action.value]
        pf = preflop + code if rnd == 0 else preflop
        fl = flop + code if rnd == 1 else flop

        if action == Action.FOLD:
            return -pot0 if player == 0 else pot1

        other_pot = pot1 if player == 0 else pot0
        new_pot0, new_pot1 = pot0, pot1

        if action == Action.RAISE:
            new_my = other_pot + BET_AMOUNTS[rnd]
            if player == 0:
                new_pot0 = new_my
            else:
                new_pot1 = new_my
            return self._br_node(
                br_hand, opp_probs, board, pf, fl,
                rnd, 1 - player, new_pot0, new_pot1, raises + 1, br_player,
            )

        # Call / Check
        my_pot = pot0 if player == 0 else pot1
        round_ended = False
        if other_pot > my_pot:
            if player == 0:
                new_pot0 = other_pot
            else:
                new_pot1 = other_pot
            round_ended = True
        elif player == 1:
            round_ended = True

        if not round_ended:
            return self._br_node(
                br_hand, opp_probs, board, pf, fl,
                rnd, 1 - player, new_pot0, new_pot1, raises, br_player,
            )

        # Round ended
        if rnd == 0:
            # Chance node: enumerate board cards, update opponent distribution
            # via P(board | br_hand, opp_hand) = remaining_cards(board) / 4
            total_val = 0.0
            for b in CARDS:
                board_weight = 0.0
                new_opp: Dict[str, float] = {}
                for h, p in opp_probs.items():
                    if p <= 0:
                        continue
                    rem = 2 - (b == br_hand) - (b == h)
                    if rem <= 0:
                        continue
                    w = p * rem / 4.0
                    new_opp[h] = w
                    board_weight += w
                if board_weight <= 0:
                    continue
                for h in new_opp:
                    new_opp[h] /= board_weight
                v = self._br_node(
                    br_hand, new_opp, b, pf, "",
                    1, 0, new_pot0, new_pot1, 0, br_player,
                )
                total_val += board_weight * v
            return total_val

        # Showdown: expected value over opponent hands
        total_val = 0.0
        for h, p in opp_probs.items():
            if p <= 0:
                continue
            if br_player == 0:
                total_val += p * _showdown(br_hand, h, board, new_pot0, new_pot1)
            else:
                total_val += p * _showdown(h, br_hand, board, new_pot0, new_pot1)
        return total_val


# ------------------------------------------------------------------
# Module-level helpers (no self, max performance)
# ------------------------------------------------------------------

def _make_key(hand: str, board: str, preflop: str, flop: str, rnd: int) -> str:
    if rnd == 0:
        return f"{hand}:{preflop}"
    return f"{hand}:{board}:{preflop}/{flop}"


def _legal_actions(raises: int) -> List[Action]:
    if raises < MAX_RAISES:
        return [Action.FOLD, Action.CALL, Action.RAISE]
    return [Action.FOLD, Action.CALL]


def _showdown(p0_hand: str, p1_hand: str, board: str,
              pot0: int, pot1: int) -> float:
    p0_pair = p0_hand == board
    p1_pair = p1_hand == board
    if p0_pair and not p1_pair:
        return pot1
    if p1_pair and not p0_pair:
        return -pot0
    if not p0_pair and not p1_pair:
        v0, v1 = CARD_VALUES[p0_hand], CARD_VALUES[p1_hand]
        if v0 > v1:
            return pot1
        if v1 > v0:
            return -pot0
    return 0  # Tie
