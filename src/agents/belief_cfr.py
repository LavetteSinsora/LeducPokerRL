"""
Belief-CFR Agent for Leduc Hold'em.

Extends the Bayesian belief framework from BeliefValueAgent, but replaces
the learned LikelihoodModel with a frozen CFR Nash equilibrium strategy
lookup for computing P(action | hand, state).

Architecture:
  - Belief tracking: P(opponent_hand) initialized from card removal,
    then updated via Bayes' rule using CFR Nash strategy as likelihood.
  - CFR strategy: Loaded from pre-trained TabularStrategyStore (frozen).
  - Value network: MLP(14 -> 64 -> 64 -> 1) with belief vector as input.
  - Action selection: 1-step lookahead (same as BeliefValueAgent).

The key insight: instead of learning an approximate P(action | hand, state)
from self-play, we use the exact Nash equilibrium policy computed by CFR.
This gives theoretically optimal likelihoods for belief updates, assuming
the opponent plays near-Nash.
"""

import numpy as np
import torch
import torch.nn as nn
from src.engine.leduc_game import Action, LeducGame
from src.engine.observation import Observation
from src.cfr.strategy import TabularStrategyStore
from .base import BaseAgent
from . import belief_common
from .belief_value import BeliefValueNetwork


class BeliefCfrAgent(BaseAgent):
    """
    Bayesian Belief Agent that uses CFR Nash equilibrium as the likelihood
    model for Bayesian belief updates over the opponent's hand.

    Instead of a learned MLP likelihood model, this agent loads a pre-trained
    CFR strategy store and looks up P(action | hand, state) directly from the
    Nash equilibrium strategy tables.
    """

    CARD_MAP = belief_common.CARD_MAP
    CARDS = belief_common.CARDS
    CARD_COUNTS = belief_common.CARD_COUNTS
    MAX_CHIPS = belief_common.MAX_CHIPS

    def __init__(self, model_path: str = None, cfr_path: str = None,
                 temperature: float = 1.0):
        self.input_size = 14  # hand(3) + board(4) + pot(2) + belief(3) + round(1) + raises(1)
        self.temperature = temperature
        self.train_mode = False

        self.model = BeliefValueNetwork(self.input_size)

        # Load CFR Nash strategy store (frozen, never trained)
        self.strategy_store = TabularStrategyStore()
        if cfr_path is None:
            cfr_path = "models/cfr_agent.pt"
        self._load_cfr_strategy(cfr_path)

        if model_path:
            self.load_model(model_path)
        self.model.eval()

    def _load_cfr_strategy(self, path: str):
        """Load the pre-trained CFR strategy store."""
        import os
        if os.path.exists(path):
            self.strategy_store.load(path)
        else:
            print(f"WARNING: CFR strategy not found at {path}. "
                  f"Belief updates will use uniform likelihoods.")

    def set_train_mode(self, mode: bool):
        self.train_mode = mode
        self.model.train(mode)
        # Note: strategy_store is frozen, never in train mode

    def save_model(self, path: str) -> None:
        torch.save({
            'value_network': self.model.state_dict(),
        }, path)

    def load_model(self, path: str) -> None:
        checkpoint = torch.load(path, weights_only=False)
        if isinstance(checkpoint, dict) and 'value_network' in checkpoint:
            self.model.load_state_dict(checkpoint['value_network'])
        else:
            # Fallback: assume it's a raw state dict
            self.model.load_state_dict(checkpoint)

    # ------------------------------------------------------------------
    # CFR-based likelihood computation
    # ------------------------------------------------------------------

    @staticmethod
    def _build_opponent_infoset_key(opponent_hand: str, board: str,
                                     current_round: int,
                                     action_history: list) -> str:
        """
        Build the CFR infoset key from the OPPONENT's perspective.

        Delegates to belief_common.build_cfr_infoset_key().
        """
        return belief_common.build_cfr_infoset_key(
            opponent_hand, board, current_round, action_history
        )

    def _get_cfr_likelihood(self, opponent_hand: str, action: Action,
                            board: str, current_round: int,
                            action_history: list,
                            legal_actions: list) -> float:
        """
        Look up P(action | opponent_hand, state) from the CFR Nash strategy.

        Args:
            opponent_hand: Hypothetical opponent hand card ('J', 'Q', or 'K').
            action: The observed action the opponent took.
            board: Current board card (or None).
            current_round: 0 (pre-flop) or 1 (flop).
            action_history: Action history up to the opponent's decision point.
            legal_actions: Legal actions at the opponent's decision point.

        Returns:
            Probability of the action under Nash strategy.
        """
        key = self._build_opponent_infoset_key(
            opponent_hand, board, current_round, action_history
        )

        # Get the Nash average strategy for this info set
        if legal_actions:
            strategy = self.strategy_store.get_average_strategy(key, legal_actions)
        else:
            # If we don't know legal actions, use all three
            strategy = self.strategy_store.get_average_strategy(
                key, [Action.FOLD, Action.CALL, Action.RAISE]
            )

        # strategy is a numpy array of shape (3,) with probs for [FOLD, CALL, RAISE]
        action_idx = int(action)
        prob = strategy[action_idx]

        # Clamp to avoid zero likelihoods (which would kill that belief entirely)
        return max(prob, 1e-6)

    # ------------------------------------------------------------------
    # Belief computation
    # ------------------------------------------------------------------

    def initialize_belief(self, my_hand: str, board: str = None) -> np.ndarray:
        """
        Initialize P(opponent_hand) from card removal logic.

        Returns:
            numpy array of shape (3,) with probabilities for [J, Q, K].
        """
        return belief_common.initialize_belief(my_hand, board)

    def update_belief(self, belief: np.ndarray, action: Action,
                      board: str, current_round: int,
                      action_history: list,
                      legal_actions: list) -> np.ndarray:
        """
        Bayesian update of belief given an observed opponent action,
        using CFR Nash strategy as the likelihood model.

        P(hand | action) proportional to P_Nash(action | hand, state) * P(hand)

        Args:
            belief: Current belief distribution, shape (3,).
            action: The action the opponent took.
            board: Board card at time of action.
            current_round: Round at time of action (0 or 1).
            action_history: Action history up to the opponent's decision.
            legal_actions: Legal actions at the opponent's decision point.

        Returns:
            Updated belief distribution, shape (3,).
        """
        likelihoods = np.zeros(3)

        for hand_idx in range(3):
            if belief[hand_idx] < 1e-8:
                continue
            hand = self.CARDS[hand_idx]
            likelihoods[hand_idx] = self._get_cfr_likelihood(
                opponent_hand=hand,
                action=action,
                board=board,
                current_round=current_round,
                action_history=action_history,
                legal_actions=legal_actions,
            )

        # Bayesian update
        posterior = belief * likelihoods
        total = posterior.sum()
        if total < 1e-10:
            return belief  # No update if everything is zero
        return posterior / total

    def compute_belief_from_history(self, obs: Observation) -> np.ndarray:
        """
        Compute the full belief vector by replaying the action history.

        Starts from the card-removal prior and applies Bayesian updates
        for each opponent action in the history, using CFR Nash likelihoods.
        """
        def _update_callback(belief, action, running_state):
            return self.update_belief(
                belief, action,
                board=running_state['board'],
                current_round=running_state['current_round'],
                action_history=running_state['action_history'],
                legal_actions=running_state['legal_actions'],
            )

        return belief_common.replay_belief_from_history(obs, _update_callback)

    # ------------------------------------------------------------------
    # Encoding
    # ------------------------------------------------------------------

    def encode_observation(self, obs: Observation, viewer_id: int = None,
                           belief: np.ndarray = None) -> torch.Tensor:
        """
        Encode observation with belief vector.

        Input vector (14 dims):
          hand(3) + board(4) + pot(2) + belief(3) + round(1) + raises(1)
        """
        if viewer_id is None:
            viewer_id = obs.current_player

        # 1. My hand (one-hot)
        hand_idx = self.CARD_MAP.get(obs.player_hand)
        hand_vec = torch.zeros(3)
        if hand_idx is not None:
            hand_vec[hand_idx] = 1.0

        # 2. Board card (one-hot): J, Q, K, None
        board_idx = self.CARD_MAP.get(obs.board, 3)
        board_vec = torch.zeros(4)
        board_vec[board_idx] = 1.0

        # 3. Pot (normalized, relative to viewer)
        p0_pot, p1_pot = obs.pot
        pot_rel = [p0_pot, p1_pot] if viewer_id == 0 else [p1_pot, p0_pot]
        pot_vec = torch.tensor(pot_rel, dtype=torch.float32) / self.MAX_CHIPS

        # 4. Belief vector (3 dims)
        if belief is None:
            belief = self.compute_belief_from_history(obs)
        belief_vec = torch.tensor(belief, dtype=torch.float32)

        # 5. Round and raises
        features = torch.tensor([
            float(obs.current_round),
            obs.raises_this_round / 2.0,
        ])

        return torch.cat([hand_vec, board_vec, pot_vec, belief_vec, features]).unsqueeze(0)

    # ------------------------------------------------------------------
    # Action selection (1-step lookahead)
    # ------------------------------------------------------------------

    def _get_value(self, obs: Observation, viewer_id: int,
                   belief: np.ndarray = None) -> float:
        encoded = self.encode_observation(obs, viewer_id=viewer_id, belief=belief)
        with torch.no_grad():
            return self.model(encoded).item()

    def get_action_evaluations(self, obs: Observation) -> list:
        """Run 1-step simulation and return predicted values with belief."""
        evaluations = []
        current_p = obs.current_player

        # Compute belief for the current state
        belief = self.compute_belief_from_history(obs)

        for action in obs.legal_actions:
            post_obs, done = LeducGame.simulate_action(obs, action)

            if done and action == Action.FOLD:
                val = -float(obs.pot[current_p])
            else:
                val = self._get_value(post_obs, viewer_id=current_p, belief=belief)

            encoded = self.encode_observation(post_obs, viewer_id=current_p, belief=belief)
            evaluations.append({
                "action": action,
                "value": val,
                "is_terminal": done,
                "encoded": encoded,
                "belief": belief.copy(),
            })
        return evaluations

    def select_action(self, obs: Observation) -> Action:
        """Select action using 1-step lookahead with belief-augmented values."""
        results = self.get_action_evaluations(obs)

        if not results:
            return Action.FOLD

        try:
            if self.train_mode:
                # Softmax (Boltzmann) exploration
                values = torch.tensor([r["value"] for r in results])
                probs = torch.softmax(values / self.temperature, dim=0)
                idx = torch.multinomial(probs, 1).item()
                return results[idx]["action"]
            else:
                # Greedy selection
                return max(results, key=lambda x: x["value"])["action"]
        except Exception as e:
            print(f"Error in BeliefCfrAgent selection: {e}")
            return results[0]["action"]
