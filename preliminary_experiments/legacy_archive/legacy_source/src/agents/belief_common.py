"""
Shared utilities for belief-based agents.

Centralizes the common infrastructure used across all belief agent variants
(belief_value, belief_cfr, belief_modulated, belief_oracle, belief_confident,
belief_stable) to eliminate duplication and ensure consistency.

Shared components:
  - Card constants (CARD_MAP, CARDS, CARD_COUNTS, MAX_CHIPS)
  - initialize_belief(): card removal prior
  - replay_belief_from_history(): game state reconstruction + belief updates
  - build_cfr_infoset_key(): CFR strategy table key construction
"""

import numpy as np
from typing import Callable, Optional, List, Tuple
from src.engine.leduc_game import Action
from src.engine.observation import Observation

# ── Card constants ────────────────────────────────────────────────

CARD_MAP = {'J': 0, 'Q': 1, 'K': 2}
CARDS = ['J', 'Q', 'K']
CARD_COUNTS = [2, 2, 2]  # 2 of each in the deck
MAX_CHIPS = 13

# Bet amounts per round (pre-flop, flop)
BET_AMOUNTS = [2, 4]


# ── Belief initialization ────────────────────────────────────────

def initialize_belief(my_hand: str, board: str = None) -> np.ndarray:
    """
    Initialize P(opponent_hand) from card removal logic.

    Given my hand and the board card, compute the prior distribution
    over the opponent's possible hand cards.

    Args:
        my_hand: My hand card ('J', 'Q', or 'K').
        board: Board card ('J', 'Q', 'K', or None if not revealed).

    Returns:
        numpy array of shape (3,) with probabilities for [J, Q, K].
    """
    counts = list(CARD_COUNTS)  # [2, 2, 2]

    # Remove my hand card from the pool
    my_idx = CARD_MAP[my_hand]
    counts[my_idx] -= 1

    # Remove board card from the pool (if revealed)
    if board is not None:
        board_idx = CARD_MAP[board]
        counts[board_idx] -= 1

    total = sum(counts)
    if total == 0:
        return np.array([1.0 / 3, 1.0 / 3, 1.0 / 3])

    return np.array(counts, dtype=np.float32) / total


# ── Game state reconstruction + belief replay ─────────────────────

def replay_belief_from_history(
    obs: Observation,
    update_fn: Callable,
) -> np.ndarray:
    """
    Compute belief by replaying action history with game state reconstruction.

    This is the core shared loop used by ALL belief agents. It:
    1. Initializes belief from card removal
    2. Walks through the action history, tracking running game state
    3. Calls update_fn(belief, action, running_state) for each opponent action
    4. Re-initializes belief at round transitions (new card removal info)

    Args:
        obs: Current observation with action_history.
        update_fn: Callback for belief updates on opponent actions.
            Signature: (belief: np.ndarray, action: Action, running_state: dict) -> np.ndarray
            running_state contains:
                - 'pot': [p0_pot, p1_pot]
                - 'current_round': 0 or 1
                - 'raises': raises this round
                - 'board': board card at this point (or None)
                - 'action_history': list of (player_id, action_name) up to this point
                - 'player_id': who acted (the opponent)
                - 'viewer_id': who is observing (us)
                - 'legal_actions': legal actions at this decision point

    Returns:
        Final belief vector, shape (3,).
    """
    my_hand = obs.player_hand
    board = obs.board

    # Initialize with card removal prior
    belief = initialize_belief(my_hand, board)

    if obs.action_history is None:
        return belief

    viewer_id = obs.current_player

    # Running game state tracking
    running_pot = [1, 1]  # antes
    running_round = 0
    running_raises = 0
    running_board = None  # Board not visible until round 1
    running_action_history = []

    for player_id, action_name in obs.action_history:
        action = Action[action_name]

        if int(player_id) != viewer_id:
            # Opponent acted — compute legal actions at this point
            if running_raises >= 2:
                legal_actions = [Action.FOLD, Action.CALL]
            else:
                legal_actions = [Action.FOLD, Action.CALL, Action.RAISE]

            # Build running state dict for the update callback
            running_state = {
                'pot': list(running_pot),
                'current_round': running_round,
                'raises': running_raises,
                'board': running_board,
                'action_history': list(running_action_history),
                'player_id': int(player_id),
                'viewer_id': viewer_id,
                'legal_actions': legal_actions,
                'my_hand': my_hand,
            }

            belief = update_fn(belief, action, running_state)

        # Record action in running history
        running_action_history.append((int(player_id), action_name))

        # Update running state based on action
        if action == Action.FOLD:
            break
        elif action == Action.CALL:
            other_player = 1 - int(player_id)
            if running_pot[other_player] > running_pot[int(player_id)]:
                running_pot[int(player_id)] = running_pot[other_player]
                # Round ends (call after raise)
                if running_round == 0:
                    running_round = 1
                    running_board = board
                    running_raises = 0
                    belief = initialize_belief(my_hand, running_board)
            else:
                # Check
                if int(player_id) == 1:
                    if running_round == 0:
                        running_round = 1
                        running_board = board
                        running_raises = 0
                        belief = initialize_belief(my_hand, running_board)
        elif action == Action.RAISE:
            bet = BET_AMOUNTS[running_round]
            other_player = 1 - int(player_id)
            running_pot[int(player_id)] = running_pot[other_player] + bet
            running_raises += 1

    return belief


# ── CFR infoset key construction ──────────────────────────────────

def build_cfr_infoset_key(
    hand: str,
    board: str,
    current_round: int,
    action_history: list,
) -> str:
    """
    Build a CFR infoset key from the acting player's perspective.

    Mirrors the key format used by CFRAgent / the CFR solver.

    Key format:
      Pre-flop: "{hand}:{preflop_actions}"
      Flop:     "{hand}:{board}:{preflop_actions}/{flop_actions}"

    Args:
        hand: The hand card of the player whose key we're building.
        board: Board card (or empty string / None).
        current_round: 0 (pre-flop) or 1 (flop).
        action_history: List of (player_id, action_name) tuples up to
            (but not including) the current decision point.

    Returns:
        String key for CFR strategy table lookup.
    """
    board_str = board if board else ""

    if not action_history:
        if current_round == 0:
            return f"{hand}:"
        return f"{hand}:{board_str}:/"

    preflop = ""
    flop = ""
    current_rnd = 0
    player = 0
    pot0, pot1 = 1, 1
    code_map = {"FOLD": "f", "CALL": "c", "RAISE": "r"}

    for _, action_name in action_history:
        code = code_map[action_name]
        if current_rnd == 0:
            preflop += code
        else:
            flop += code

        if action_name == "FOLD":
            break

        if action_name == "RAISE":
            other_pot = pot1 if player == 0 else pot0
            new_my = other_pot + BET_AMOUNTS[current_rnd]
            if player == 0:
                pot0 = new_my
            else:
                pot1 = new_my
            player = 1 - player
        else:  # CALL/CHECK
            other_pot = pot1 if player == 0 else pot0
            my_pot = pot0 if player == 0 else pot1
            round_ended = False
            if other_pot > my_pot:
                if player == 0:
                    pot0 = other_pot
                else:
                    pot1 = other_pot
                round_ended = True
            elif player == 1:
                round_ended = True

            if round_ended and current_rnd == 0:
                current_rnd = 1
                player = 0
            elif not round_ended:
                player = 1 - player

    if current_round == 0:
        return f"{hand}:{preflop}"
    return f"{hand}:{board_str}:{preflop}/{flop}"
