"""
CFR Agent — inference-time wrapper for a trained CFR strategy.

Uses the average (converged Nash) strategy from a TabularStrategyStore
to play Leduc Hold'em. The CFR training is done by LeducCFRSolver;
this agent simply looks up and samples from the resulting strategy.
"""

import random

from agents.base import BaseAgent
from engine.leduc_game import Action
from engine.observation import Observation

from .strategy import TabularStrategyStore

BET_AMOUNTS = [2, 4]  # Pre-flop, Flop — needed for round boundary detection


class CFRAgent(BaseAgent):
    """Agent that plays the Nash equilibrium strategy computed by CFR.

    At each decision point, converts the game observation into an
    information set key and samples from the average strategy.
    """

    def __init__(self, model_path: str = None):
        self.strategy_store = TabularStrategyStore()
        if model_path:
            self.load_model(model_path)

    def select_action(self, obs: Observation) -> Action:
        """Sample an action from the Nash equilibrium strategy."""
        key = self._obs_to_key(obs)
        strategy = self.strategy_store.get_average_strategy(key, obs.legal_actions)

        choices = []
        weights = []
        for action in obs.legal_actions:
            choices.append(action)
            weights.append(strategy[action.value])

        return random.choices(choices, weights=weights, k=1)[0]

    def get_action_evaluations(self, obs: Observation) -> list:
        """Return per-action probabilities for the analyzer UI."""
        key = self._obs_to_key(obs)
        strategy = self.strategy_store.get_average_strategy(key, obs.legal_actions)
        return [
            {"action": a, "probability": strategy[a.value]}
            for a in obs.legal_actions
        ]

    def save_model(self, path: str) -> None:
        self.strategy_store.save(path)

    def load_model(self, path: str) -> None:
        self.strategy_store.load(path)

    @staticmethod
    def _obs_to_key(obs: Observation) -> str:
        """Convert Observation to CFR infoset key.

        Replays the action history to determine the preflop/flop boundary,
        then builds the key in the same format the solver uses:
            Pre-flop: "{hand}:{preflop_actions}"
            Flop:     "{hand}:{board}:{preflop_actions}/{flop_actions}"
        """
        hand = obs.player_hand
        board = obs.board if obs.board else ""
        rnd = obs.current_round

        if not obs.action_history:
            if rnd == 0:
                return f"{hand}:"
            return f"{hand}:{board}:/"

        preflop = ""
        flop = ""
        current_rnd = 0
        player = 0
        pot0, pot1 = 1, 1
        code_map = {"FOLD": "f", "CALL": "c", "RAISE": "r"}

        for _, action_name in obs.action_history:
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

        if rnd == 0:
            return f"{hand}:{preflop}"
        return f"{hand}:{board}:{preflop}/{flop}"
