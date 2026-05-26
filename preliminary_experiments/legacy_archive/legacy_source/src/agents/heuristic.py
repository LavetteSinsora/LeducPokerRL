"""
Strong Heuristic Agent for Leduc Hold'em (Aleta Kodum).

This agent implements a comprehensive rule-based strategy that considers:
- Hand strength (pair vs high card)
- Position (acting first vs second)
- Pot odds and betting round
- Opponent aggression patterns
- Board texture and outs
"""

from src.engine.leduc_game import Action
from src.engine.observation import Observation
from .base import BaseAgent

# Card rankings: K > Q > J
CARD_RANKS = {'J': 0, 'Q': 1, 'K': 2}


class HeuristicAgent(BaseAgent):
    """
    A strong rule-based agent for Leduc Hold'em.
    
    Strategy principles:
    1. Always raise with pairs (nuts or near-nuts)
    2. Value bet strong hands, bluff occasionally with weak hands
    3. Consider pot odds when calling
    4. Adjust based on position and betting patterns
    """
    
    def encode_observation(self, obs: Observation):
        return obs

    def select_action(self, obs: Observation) -> Action:
        """Select the best action based on heuristic rules."""
        hand = obs.player_hand
        board = obs.board
        legal = obs.legal_actions
        pot = obs.pot
        current_round = obs.current_round
        current_player = obs.current_player
        
        # Calculate hand strength
        has_pair = board is not None and hand == board
        hand_rank = CARD_RANKS.get(hand, -1)
        
        # Calculate pot odds context
        my_pot = pot[current_player]
        opp_pot = pot[1 - current_player]
        pot_total = my_pot + opp_pot
        to_call = opp_pot - my_pot
        facing_raise = to_call > 0
        
        # Determine betting amount for this round
        bet_amount = 2 if current_round == 0 else 4
        
        # Position indicator: 0 = acting first, 1 = acting second in round
        # After raises, we know opponent has shown strength
        
        # =================================================================
        # PRE-FLOP STRATEGY (Round 0) - No community card yet
        # =================================================================
        if current_round == 0:
            return self._preflop_strategy(
                hand, hand_rank, legal, facing_raise, pot_total, to_call, current_player
            )
        
        # =================================================================
        # FLOP STRATEGY (Round 1) - Community card is visible
        # =================================================================
        else:
            return self._flop_strategy(
                hand, hand_rank, board, has_pair, legal, 
                facing_raise, pot_total, to_call, bet_amount, current_player
            )
    
    def _preflop_strategy(self, hand: str, hand_rank: int, legal: list, 
                          facing_raise: bool, pot_total: int, to_call: int,
                          current_player: int) -> Action:
        """
        Pre-flop strategy without community card information.
        
        Key insights:
        - Kings have ~56% equity vs random hand
        - Queens have ~50% equity vs random hand  
        - Jacks have ~44% equity vs random hand
        - Position matters: acting second has information advantage
        """
        
        # FACING A RAISE
        if facing_raise:
            if hand == 'K':
                # Kings are strong enough to re-raise or call a raise
                if Action.RAISE in legal:
                    return Action.RAISE
                return Action.CALL
            elif hand == 'Q':
                # Queens call a raise (decent equity), but don't re-raise
                return Action.CALL
            else:  # Jack
                # Jacks have poor equity against a raising range
                # Fold if pot odds are bad, otherwise call if getting good odds
                if pot_total > 0 and to_call / (pot_total + to_call) < 0.35:
                    return Action.CALL
                return Action.FOLD
        
        # NOT FACING A RAISE (acting first or opponent checked)
        if hand == 'K':
            # Always raise with Kings for value
            if Action.RAISE in legal:
                return Action.RAISE
            return Action.CALL
        elif hand == 'Q':
            # Queens: raise as a thin value bet / to take initiative
            # Also serves as a balance with our King raises
            if Action.RAISE in legal:
                return Action.RAISE
            return Action.CALL
        else:  # Jack
            # Jacks: mostly check/call to realize equity and occasionally bluff
            # Bluff ~20% of the time to balance our raising range
            import random
            if Action.RAISE in legal and random.random() < 0.20:
                return Action.RAISE
            return Action.CALL
    
    def _flop_strategy(self, hand: str, hand_rank: int, board: str, has_pair: bool,
                       legal: list, facing_raise: bool, pot_total: int, 
                       to_call: int, bet_amount: int, current_player: int) -> Action:
        """
        Post-flop strategy with community card visible.
        
        Key insights:
        - Pairs are very strong (only lose to higher pair)
        - High cards matter when no pair
        - Bluffing becomes important for balance
        - Pot odds calculation is crucial
        """
        board_rank = CARD_RANKS.get(board, -1)
        
        # =================================================================
        # PAIR HANDS - Very strong, extract maximum value
        # =================================================================
        if has_pair:
            # We have a pair - this is the nuts or near-nuts
            if hand == 'K':
                # Kings pair is the absolute nuts - always raise
                if Action.RAISE in legal:
                    return Action.RAISE
                return Action.CALL
            elif hand == 'Q':
                # Queens pair - very strong, only loses to K pair
                # Raise for value, but be aware K pair exists
                if Action.RAISE in legal:
                    return Action.RAISE
                return Action.CALL
            else:  # Jack pair
                # Jack pair - loses to Q pair and K pair
                # Still strong enough to raise for value most of the time
                if Action.RAISE in legal:
                    return Action.RAISE
                return Action.CALL
        
        # =================================================================
        # NO PAIR - Play based on high card and position
        # =================================================================
        
        # Calculate our potential to make a pair (there's still 1 copy of our card)
        # But in Leduc, the board is final - no more cards coming
        # So high card is our hand strength
        
        if facing_raise:
            # Opponent is showing strength
            return self._respond_to_flop_raise(
                hand, hand_rank, board, board_rank, pot_total, to_call, legal
            )
        
        # NOT FACING A RAISE
        if hand == 'K':
            # King high is strong - board can only pair with Q or J
            if board == 'K':
                # We can't have K if board is K and we don't have pair
                # This shouldn't happen, but just in case
                return Action.CALL
            # King high when board is Q or J - we might have best hand
            if Action.RAISE in legal:
                return Action.RAISE
            return Action.CALL
        
        elif hand == 'Q':
            # Queen high - decent but vulnerable
            if board == 'K':
                # Board is K, opponent might have K pair or K high
                # Check/call is safer
                return Action.CALL
            elif board == 'J':
                # We beat J, only K beats us
                # Thin value bet
                if Action.RAISE in legal:
                    return Action.RAISE
                return Action.CALL
            else:  # board == 'Q' - we'd have a pair, handled above
                return Action.CALL
        
        else:  # Jack
            # Jack high is weak
            if board == 'K' or board == 'Q':
                # We only beat if opponent also has J
                # Consider bluffing occasionally (representing a pair)
                import random
                if Action.RAISE in legal and random.random() < 0.25:
                    # Bluff raise - representing we have the board card
                    return Action.RAISE
                # Otherwise check/call to catch opponent's bluffs
                return Action.CALL
            else:  # board == 'J' - would be a pair, handled above
                return Action.CALL
    
    def _respond_to_flop_raise(self, hand: str, hand_rank: int, board: str,
                                board_rank: int, pot_total: int, to_call: int,
                                legal: list) -> Action:
        """
        Decision making when facing a raise on the flop.
        Uses pot odds and hand strength to determine response.
        """
        # Calculate pot odds - what percentage of pot we need to call
        pot_odds = to_call / (pot_total + to_call) if (pot_total + to_call) > 0 else 0
        
        # Estimate our equity against opponent's raising range
        # Raising range on flop likely contains: pairs, and strong high cards
        
        if hand == 'K':
            # King high against a raise - might be good
            if board != 'K':
                # Opponent might have K pair (if board isn't K)
                # Or be bluffing, or value betting a weaker pair
                # Call to catch bluffs and beat weaker hands
                return Action.CALL
            else:
                # Board is K, we'd have a pair - shouldn't reach here
                return Action.CALL
        
        elif hand == 'Q':
            # Queen high facing a raise
            if board == 'J':
                # Q high on J board - beats J high and J pair... wait no
                # Opponent with J pair beats us
                # Fold to aggression unless pot odds are great
                if pot_odds < 0.25:
                    return Action.CALL
                return Action.FOLD
            elif board == 'K':
                # Marginal situation - opponent likely has K pair or K high
                # Getting bluffed sometimes, fold is reasonable
                if pot_odds < 0.20:
                    return Action.CALL
                return Action.FOLD
            else:  # board == 'Q' - we'd have pair
                return Action.CALL
        
        else:  # Jack
            # Jack high facing a raise is very weak
            # We only beat bluffs
            # Need very good pot odds to call
            if pot_odds < 0.18:
                return Action.CALL
            return Action.FOLD
