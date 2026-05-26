"""
Belief Oracle Agent for Leduc Hold'em.

Learns a PERFECT-INFORMATION value function V(s, my_hand, opp_hand) during
self-play training where both hands are known.  At decision time (both
training and evaluation), the agent weights across possible opponent hands
using its current belief distribution:

    value(action) = sum_h  belief(h) * V(post_state(obs, action), my_hand, h)

This cleanly separates:
  - "Learning accurate values" -- trained with ground truth opponent hand
  - "Handling uncertainty"     -- belief weighting only at inference

Architecture:
  - OracleValueNetwork: MLP(14 -> 64 -> 64 -> 1)
      Input: hand(3) + board(4) + pot(2) + opp_hand(3) + round(1) + raises(1)
  - Belief tracking: same as BeliefValueAgent (card removal init, Bayes' update)
  - Likelihood source: configurable (default: CFR Nash)
"""

import os
import numpy as np
import torch
import torch.nn as nn
from src.engine.leduc_game import Action, LeducGame
from src.engine.observation import Observation
from .base import BaseAgent
from . import belief_common


class OracleValueNetwork(nn.Module):
    """
    Perfect-information value network: V(state, my_hand, opp_hand).

    Input (14 dims):
      hand(3 one-hot) + board(4 one-hot incl None) + pot(2 normalized)
      + opp_hand(3 one-hot) + round(1) + raises(1)
    Output: scalar V(s)
    """
    def __init__(self, input_size: int = 14, hidden_size: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class BeliefOracleAgent(BaseAgent):
    """
    Belief Oracle Agent -- perfect-info value function with belief-weighted
    action selection.

    During training (self-play), the value network sees the TRUE opponent hand.
    During action selection (BOTH training and evaluation), actions are chosen
    by weighting the value network's predictions across possible opponent hands
    according to the current belief distribution.
    """

    CARD_MAP = belief_common.CARD_MAP
    CARDS = belief_common.CARDS
    CARD_COUNTS = belief_common.CARD_COUNTS
    MAX_CHIPS = belief_common.MAX_CHIPS

    def __init__(self, model_path: str = None, temperature: float = 1.0,
                 likelihood_source: str = 'cfr_nash'):
        """
        Args:
            model_path: Path to saved model checkpoint.
            temperature: Boltzmann temperature for exploration during training.
            likelihood_source: Source for belief-update likelihoods.
                'cfr_nash' -- use trained CFR Nash equilibrium strategy
                'uniform'  -- uniform likelihoods (no belief update)
        """
        self.input_size = 14  # hand(3)+board(4)+pot(2)+opp_hand(3)+round(1)+raises(1)
        self.temperature = temperature
        self.train_mode = False
        self.likelihood_source = likelihood_source

        self.model = OracleValueNetwork(self.input_size)

        # CFR strategy store for Nash likelihoods (loaded lazily)
        self._cfr_store = None

        if model_path:
            self.load_model(model_path)
        self.model.eval()

    # ------------------------------------------------------------------
    # Model persistence
    # ------------------------------------------------------------------

    def set_train_mode(self, mode: bool):
        self.train_mode = mode
        self.model.train(mode)

    def save_model(self, path: str) -> None:
        torch.save({
            'oracle_value_network': self.model.state_dict(),
        }, path)

    def load_model(self, path: str) -> None:
        checkpoint = torch.load(path, weights_only=False)
        if isinstance(checkpoint, dict) and 'oracle_value_network' in checkpoint:
            self.model.load_state_dict(checkpoint['oracle_value_network'])
        else:
            # Fallback: raw state dict
            self.model.load_state_dict(checkpoint)

    # ------------------------------------------------------------------
    # CFR Nash likelihood source
    # ------------------------------------------------------------------

    def _get_cfr_store(self):
        """Lazily load the CFR strategy store."""
        if self._cfr_store is None:
            from src.cfr.strategy import TabularStrategyStore
            store = TabularStrategyStore()
            cfr_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                'models', 'cfr_agent.pt'
            )
            if os.path.exists(cfr_path):
                store.load(cfr_path)
                self._cfr_store = store
            else:
                # Fallback: empty store (will give uniform strategies)
                self._cfr_store = store
        return self._cfr_store

    def _get_cfr_action_probs(self, obs: Observation) -> np.ndarray:
        """
        Get action probabilities from the CFR Nash equilibrium strategy
        for the given observation.

        Returns:
            np.ndarray of shape (3,) -- P(FOLD), P(CALL), P(RAISE)
        """
        from src.agents.cfr_agent import CFRAgent
        key = CFRAgent._obs_to_key(obs)
        store = self._get_cfr_store()
        legal_actions = obs.legal_actions if obs.legal_actions else [Action.FOLD, Action.CALL]
        strategy = store.get_average_strategy(key, legal_actions)
        return strategy  # shape (3,)

    # ------------------------------------------------------------------
    # Belief computation (same logic as BeliefValueAgent)
    # ------------------------------------------------------------------

    def initialize_belief(self, my_hand: str, board: str = None) -> np.ndarray:
        """
        Initialize P(opponent_hand) from card removal logic.

        Returns:
            numpy array of shape (3,) with probabilities for [J, Q, K].
        """
        return belief_common.initialize_belief(my_hand, board)

    def update_belief(self, belief: np.ndarray, action: Action,
                      obs: Observation, actor_id: int) -> np.ndarray:
        """
        Bayesian update of belief given an observed opponent action.

        Uses CFR Nash likelihoods: for each possible opponent hand h,
        compute P(action | h, game_state) from the CFR equilibrium strategy,
        then apply Bayes' rule.

        Args:
            belief: Current belief distribution, shape (3,).
            action: The action the opponent took.
            obs: The observation at the time the opponent acted.
            actor_id: The player who took the action.

        Returns:
            Updated belief distribution, shape (3,).
        """
        if self.likelihood_source == 'uniform':
            return belief  # No update with uniform likelihoods

        action_idx = int(action)
        likelihoods = np.zeros(3)

        for hand_idx in range(3):
            if belief[hand_idx] < 1e-8:
                continue

            # Build a hypothetical observation as if the opponent held this hand
            hyp_obs = Observation(
                player_hand=self.CARDS[hand_idx],
                board=obs.board,
                pot=list(obs.pot),
                current_player=actor_id,
                current_round=obs.current_round,
                legal_actions=obs.legal_actions if obs.legal_actions else [Action.FOLD, Action.CALL],
                is_finished=False,
                raises_this_round=obs.raises_this_round,
                action_history=obs.action_history,
            )

            # Get Nash action probabilities for this hypothetical hand
            action_probs = self._get_cfr_action_probs(hyp_obs)
            likelihoods[hand_idx] = action_probs[action_idx]

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
        for each opponent action in the history.
        """
        def _update_callback(belief, action, running_state):
            hist_obs = Observation(
                player_hand=running_state['my_hand'],
                board=running_state['board'],
                pot=running_state['pot'],
                current_player=running_state['player_id'],
                current_round=running_state['current_round'],
                legal_actions=running_state['legal_actions'],
                is_finished=False,
                raises_this_round=running_state['raises'],
            )
            return self.update_belief(belief, action, hist_obs,
                                      actor_id=running_state['player_id'])

        return belief_common.replay_belief_from_history(obs, _update_callback)

    # ------------------------------------------------------------------
    # Encoding
    # ------------------------------------------------------------------

    def encode_state_with_opp(self, obs: Observation, viewer_id: int,
                              opp_hand_idx: int) -> torch.Tensor:
        """
        Encode observation with a SPECIFIC opponent hand (perfect info).

        Input vector (14 dims):
          hand(3) + board(4) + pot(2) + opp_hand(3) + round(1) + raises(1)

        Args:
            obs: Game observation.
            viewer_id: Which player's perspective.
            opp_hand_idx: Index of opponent hand (0=J, 1=Q, 2=K).

        Returns:
            Tensor of shape (1, 14).
        """
        # 1. My hand (one-hot, 3 dims)
        hand_idx = self.CARD_MAP.get(obs.player_hand)
        hand_vec = torch.zeros(3)
        if hand_idx is not None:
            hand_vec[hand_idx] = 1.0

        # 2. Board card (one-hot, 4 dims: J, Q, K, None)
        board_idx = self.CARD_MAP.get(obs.board, 3)
        board_vec = torch.zeros(4)
        board_vec[board_idx] = 1.0

        # 3. Pot (normalized, relative to viewer, 2 dims)
        p0_pot, p1_pot = obs.pot
        pot_rel = [p0_pot, p1_pot] if viewer_id == 0 else [p1_pot, p0_pot]
        pot_vec = torch.tensor(pot_rel, dtype=torch.float32) / self.MAX_CHIPS

        # 4. Opponent hand (one-hot, 3 dims)
        opp_vec = torch.zeros(3)
        opp_vec[opp_hand_idx] = 1.0

        # 5. Round and raises (2 dims)
        features = torch.tensor([
            float(obs.current_round),
            obs.raises_this_round / 2.0,
        ])

        return torch.cat([hand_vec, board_vec, pot_vec, opp_vec, features]).unsqueeze(0)

    def encode_post_state_partial(self, obs: Observation,
                                  viewer_id: int) -> list:
        """
        Encode the non-opponent-hand portion of the state.

        Returns the list of floats for hand(3)+board(4)+pot(2)+round(1)+raises(1) = 11 dims.
        The caller will append opp_hand(3) to get the full 14-dim input.
        """
        # 1. My hand
        hand_idx = self.CARD_MAP.get(obs.player_hand)
        hand_vec = [0.0, 0.0, 0.0]
        if hand_idx is not None:
            hand_vec[hand_idx] = 1.0

        # 2. Board card
        board_idx = self.CARD_MAP.get(obs.board, 3)
        board_vec = [0.0, 0.0, 0.0, 0.0]
        board_vec[board_idx] = 1.0

        # 3. Pot (normalized, relative to viewer)
        p0_pot, p1_pot = obs.pot
        pot_rel = [p0_pot, p1_pot] if viewer_id == 0 else [p1_pot, p0_pot]
        pot_vec = [p / self.MAX_CHIPS for p in pot_rel]

        # 4. Round and raises
        features = [float(obs.current_round), obs.raises_this_round / 2.0]

        return hand_vec + board_vec + pot_vec + features

    # ------------------------------------------------------------------
    # Action selection (belief-weighted 1-step lookahead)
    # ------------------------------------------------------------------

    def _belief_weighted_value(self, obs: Observation, viewer_id: int,
                               belief: np.ndarray) -> float:
        """
        Compute belief-weighted value for a given post-action state.

        value = sum_h belief(h) * V(obs, my_hand, h)
        """
        partial = self.encode_post_state_partial(obs, viewer_id)
        weighted_value = 0.0

        with torch.no_grad():
            for h_idx in range(3):
                if belief[h_idx] < 1e-8:
                    continue

                opp_onehot = [0.0, 0.0, 0.0]
                opp_onehot[h_idx] = 1.0

                # Insert opp_hand after pot (position 9) and before round/raises
                # Full order: hand(3) + board(4) + pot(2) + opp_hand(3) + round(1) + raises(1)
                full_enc = partial[:9] + opp_onehot + partial[9:]
                inp = torch.tensor(full_enc, dtype=torch.float32).unsqueeze(0)
                v = self.model(inp).item()
                weighted_value += belief[h_idx] * v

        return weighted_value

    def get_action_evaluations(self, obs: Observation) -> list:
        """
        Run 1-step simulation and return belief-weighted values for each action.
        """
        evaluations = []
        current_p = obs.current_player

        # Compute belief for the current state
        belief = self.compute_belief_from_history(obs)

        for action in obs.legal_actions:
            post_obs, done = LeducGame.simulate_action(obs, action)

            if done and action == Action.FOLD:
                val = -float(obs.pot[current_p])
            else:
                val = self._belief_weighted_value(
                    post_obs, viewer_id=current_p, belief=belief
                )

            evaluations.append({
                "action": action,
                "value": val,
                "is_terminal": done,
                "belief": belief.copy(),
            })
        return evaluations

    def select_action(self, obs: Observation) -> Action:
        """
        Select action using belief-weighted 1-step lookahead.

        Both training and evaluation use the same selection mechanism:
        belief-weighted values. The only difference is exploration:
        - Training: Boltzmann (softmax) exploration with temperature
        - Evaluation: Greedy
        """
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
            print(f"Error in BeliefOracleAgent selection: {e}")
            return results[0]["action"]
