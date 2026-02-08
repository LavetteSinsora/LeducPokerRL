import random
import copy
from enum import IntEnum
from .observation import Observation

class Action(IntEnum):
    FOLD = 0
    CALL = 1
    RAISE = 2

class LeducGame:
    """
    Leduc Hold'em Engine.
    Rules:
    - 6 cards: 2x Jack, 2x Queen, 2x King.
    - 2 players.
    - 2 rounds: Pre-flop and Flop.
    - 1 community card dealt at the start of Flop.
    - Max 2 raises per round.
    - Betting: Pre-flop chips = 2, Flop chips = 4.
    """
    
    CARDS = ['J', 'J', 'Q', 'Q', 'K', 'K']
    BET_AMOUNTS = [2, 4]  # Pre-flop, Flop
    MAX_RAISES = 2

    def __init__(self):
        self.reset()

    def reset(self):
        """Resets the game to initial state."""
        self.deck = list(self.CARDS)
        random.shuffle(self.deck)
        
        self.player_hands = [self.deck.pop(), self.deck.pop()]
        self.board = None
        self.pot = [1, 1]  # Ante
        self.current_round = 0  # 0: Pre-flop, 1: Flop
        self.current_player = 0
        self.raises_this_round = 0
        self.is_finished = False
        self.winner = None
        self.history = []
        
        # Round state
        self.round_betting_ended = False
        self.last_action_was_raise = False
        
        return self.get_observation()

    def set_state(self, observation: Observation):
        """Sets the game state from an Observation object (for simulation)."""
        self.current_round = observation.current_round
        self.pot = list(observation.pot)
        self.current_player = observation.current_player
        self.board = observation.board
        self.is_finished = observation.is_finished
        self.winner = None # Reset winner for simulation
        
        # We assume player_hand is for the current acting player.
        self.player_hands[self.current_player] = observation.player_hand
        
        # Note: history and other internal states aren't perfectly restored from obs,
        # but for 1-step lookahead this is sufficient.
        # Opponent hand is left as is or UNKNOWN.
        other_player = 1 - self.current_player
        if self.player_hands[other_player] not in self.CARDS:
            self.player_hands[other_player] = 'UNKNOWN'
        
    def copy(self):
        """Returns a deep copy of the game instance."""
        return copy.deepcopy(self)

    def step(self, action):
        """Executes an action."""
        if self.is_finished:
            raise ValueError("Game is already finished.")
        
        player = self.current_player
        bet_amount = self.BET_AMOUNTS[self.current_round]
        
        if action == Action.FOLD:
            self.history.append((player, "FOLD"))
            self.is_finished = True
            self.winner = 1 - player
            return self.get_observation(), self.get_reward(), True, {}
            
        elif action == Action.CALL:
            self.history.append((player, "CALL"))
            # If the other player raised, we match. 
            # If no one raised, it's a check.
            other_player_pot = self.pot[1 - player]
            if other_player_pot > self.pot[player]:
                self.pot[player] = other_player_pot
                self.round_betting_ended = True
            else:
                # CHECK
                if player == 1: # Second player checks, round ends
                    self.round_betting_ended = True
            
        elif action == Action.RAISE:
            if self.raises_this_round >= self.MAX_RAISES:
                raise ValueError("Max raises reached.")
            
            self.history.append((player, "RAISE"))
            # To raise: match opponent + add bet_amount
            other_player_pot = self.pot[1 - player]
            self.pot[player] = other_player_pot + bet_amount
            self.raises_this_round += 1
            self.last_action_was_raise = True
            # When someone raises, the other player MUST act again
            self.round_betting_ended = False

        # Switch player
        self.current_player = 1 - self.current_player
        
        # Check round end
        if self.round_betting_ended:
            if self.current_round == 0:
                self._transition_to_flop()
            else:
                self._showdown()
        
        return self.get_observation(), self.get_reward(), self.is_finished, {}

    def _transition_to_flop(self):
        """Moves from Pre-flop to Flop."""
        self.current_round = 1
        self.board = self.deck.pop()
        self.current_player = 0
        self.raises_this_round = 0
        self.round_betting_ended = False
        self.last_action_was_raise = False

    def _showdown(self):
        """Ends game and determines winner."""
        self.is_finished = True
        p1_hand = self.player_hands[0]
        p2_hand = self.player_hands[1]
        
        # Guard against incomplete information during simulation
        if p1_hand == 'UNKNOWN' or p2_hand == 'UNKNOWN':
            self.winner = -2 # Incomplete info
            return
            
        score1 = self._evaluate_hand(p1_hand)
        score2 = self._evaluate_hand(p2_hand)
        
        if score1 > score2:
            self.winner = 0
        elif score2 > score1:
            self.winner = 1
        else:
            self.winner = -1  # Tie

    def _evaluate_hand(self, player_card):
        """Returns hand strength score."""
        card_values = {'J': 0, 'Q': 1, 'K': 2}
        
        # Tie-breaker: Pair > High Card
        if player_card == self.board:
            return 10 + card_values[player_card]
        return card_values[player_card]

    def get_reward(self):
        """Returns rewards for players. Only meaningful when finished."""
        if not self.is_finished:
            return [0, 0]
        
        if self.winner == 0:
            return [self.pot[1], -self.pot[1]]
        elif self.winner == 1:
            return [-self.pot[0], self.pot[0]]
        elif self.winner == -1:
            return [0, 0]
        else: # winner == -2 (incomplete info)
            return [0, 0]

    def get_observation(self, viewer_id=None) -> Observation:
        """Returns the observation from a specific player's perspective."""
        if viewer_id is None:
            viewer_id = self.current_player
            
        # The observation focused on the hand of the CURRENT acting player.
        # This hand is only visible to the viewer if the viewer IS that player.
        acting_player_hand = self.player_hands[self.current_player]
        if viewer_id != self.current_player:
            acting_player_hand = 'UNKNOWN'
            
        return Observation(
            player_hand=acting_player_hand,
            board=self.board,
            pot=list(self.pot),
            current_player=self.current_player,
            current_round=self.current_round,
            legal_actions=self._get_legal_actions(),
            is_finished=self.is_finished
        )

    def _get_legal_actions(self):
        actions = [Action.FOLD, Action.CALL]
        if self.raises_this_round < self.MAX_RAISES:
            actions.append(Action.RAISE)
        return actions

    def get_legal_actions(self):
        """Returns the list of legal actions for the current state."""
        return self._get_legal_actions()

    def __repr__(self):
        return f"LeducGame(Round={self.current_round}, Pot={self.pot}, Board={self.board}, Hands={self.player_hands})"

if __name__ == "__main__":
    # Quick manual run
    game = LeducGame()
    obs = game.reset()
    print("Start:", obs)
    while not game.is_finished:
        action = random.choice(obs["legal_actions"])
        obs, reward, done, _ = game.step(action)
        print(f"Action: {action.name}, Reward: {reward}, Obs: {obs}")
