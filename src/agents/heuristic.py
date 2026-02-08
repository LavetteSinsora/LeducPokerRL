from src.engine.leduc_game import Action
from src.engine.observation import Observation
from .base import BaseAgent

class HeuristicAgent(BaseAgent):
    """
    A simple rule-based agent.
    """
    def encode_observation(self, obs: Observation):
        return obs

    def select_action(self, obs: Observation):
        hand = obs.player_hand
        board = obs.board
        legal = obs.legal_actions
        
        # Simple logic: raise if King or pair
        if Action.RAISE in legal:
            if hand == 'K' or (board and hand == board):
                return Action.RAISE
        if Action.CALL in legal:
            return Action.CALL
        return Action.FOLD
