"""
Belief-Modulated Agent for Leduc Hold'em.

Combines CFR Nash equilibrium as a base likelihood model with a learned
gated modulation layer that adapts based on opponent macro statistics.

Architecture:
  - Belief tracking: P(opponent_hand) initialized from card removal,
    updated via Bayes' rule using CFR Nash + learned modulation.
  - Likelihood model:
      log P_adjusted(a | h, state) = log pi_Nash(a | h, state) + gate(opp_stats) * delta(opp_stats)
    Where:
      pi_Nash: frozen CFR strategy tables (average/Nash strategy)
      gate: MLP(4 -> 16 -> 1 -> sigmoid) -- opponent stats -> scalar in [0,1]
      delta: MLP(4 -> 16 -> 3) -- opponent stats -> per-action logit adjustments
      opp_stats = [fold_rate, call_rate, raise_rate, aggression_factor]
  - Value network: MLP(14 -> 64 -> 64 -> 1), same as BeliefValueAgent.
  - Action selection: 1-step lookahead with belief-augmented values.

The modulation is STATE-AGNOSTIC: the gate and delta receive only opponent
macro statistics, not the current game state.  This is intentional -- with
~30 hands of opponent data there is not enough information for state-conditional
modulation.  The Nash base already provides state-conditional behavior; the
modulation captures "this opponent is generally more aggressive/passive than Nash."
"""

import numpy as np
import torch
import torch.nn as nn
from dataclasses import replace
from src.engine.leduc_game import Action, LeducGame
from src.engine.observation import Observation
from src.cfr.strategy import TabularStrategyStore
from .base import BaseAgent
from . import belief_common
from .belief_value import BeliefValueNetwork


class ModulationGate(nn.Module):
    """
    Confidence gate that controls modulation strength.

    Takes opponent macro stats (4 dims) and outputs a scalar in [0, 1].
    Outputs low values early in a session (stats unreliable) and higher
    values once enough opponent data has been accumulated.
    """
    def __init__(self, input_size: int = 4, hidden_size: int = 16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ModulationDelta(nn.Module):
    """
    Per-action logit adjustment network.

    Takes opponent macro stats (4 dims) and outputs 3 logit adjustments
    (one per action: FOLD, CALL, RAISE).
    """
    def __init__(self, input_size: int = 4, hidden_size: int = 16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 3),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class BeliefModulatedAgent(BaseAgent):
    """
    Bayesian Belief Agent with CFR Nash base + learned modulation.

    Maintains a belief distribution over the opponent's hand card,
    updated within each hand using:
      adjusted_probs = softmax( log(pi_Nash) + gate(stats) * delta(stats) )

    The value network receives belief-augmented observations and uses
    1-step lookahead for action selection.
    """

    CARD_MAP = belief_common.CARD_MAP
    CARDS = belief_common.CARDS
    CARD_COUNTS = belief_common.CARD_COUNTS
    MAX_CHIPS = belief_common.MAX_CHIPS

    def __init__(self, model_path: str = None, cfr_model_path: str = None,
                 temperature: float = 1.0):
        self.input_size = 14  # hand(3) + board(4) + pot(2) + belief(3) + round(1) + raises(1)
        self.temperature = temperature
        self.train_mode = False

        # Value network (same architecture as BeliefValueAgent)
        self.model = BeliefValueNetwork(self.input_size)

        # CFR strategy store (frozen Nash base)
        self.strategy_store = TabularStrategyStore()
        if cfr_model_path:
            self.strategy_store.load(cfr_model_path)

        # Modulation networks
        self.gate_net = ModulationGate(input_size=4, hidden_size=16)
        self.delta_net = ModulationDelta(input_size=4, hidden_size=16)

        if model_path:
            self.load_model(model_path)
        self.model.eval()
        self.gate_net.eval()
        self.delta_net.eval()

    def set_train_mode(self, mode: bool):
        self.train_mode = mode
        self.model.train(mode)
        self.gate_net.train(mode)
        self.delta_net.train(mode)
        # Strategy store is always frozen -- no parameters to toggle

    def save_model(self, path: str) -> None:
        torch.save({
            'value_network': self.model.state_dict(),
            'gate_net': self.gate_net.state_dict(),
            'delta_net': self.delta_net.state_dict(),
        }, path)

    def load_model(self, path: str) -> None:
        checkpoint = torch.load(path, weights_only=False)
        self.model.load_state_dict(checkpoint['value_network'])
        self.gate_net.load_state_dict(checkpoint['gate_net'])
        self.delta_net.load_state_dict(checkpoint['delta_net'])

    # ------------------------------------------------------------------
    # Opponent stats encoding
    # ------------------------------------------------------------------

    def _encode_opp_stats(self, obs: Observation) -> torch.Tensor:
        """
        Extract 4-dim opponent macro stats from the observation.

        Returns [fold_rate, call_rate, raise_rate, aggression_factor].

        When no opponent_stats are available, returns uninformative defaults:
        [0.33, 0.33, 0.33, 0.0]  (uniform actions, zero aggression).
        """
        if obs.opponent_stats is not None and hasattr(obs.opponent_stats, 'total_actions'):
            stats = obs.opponent_stats
            if stats.total_actions > 0:
                fold_rate = stats.fold_count / stats.total_actions
                call_rate = stats.call_count / stats.total_actions
                raise_rate = stats.raise_count / stats.total_actions
                # Aggression factor: (raise + call) > 0 ? raise / (raise + call) : 0
                denom = stats.raise_count + stats.call_count
                aggression = stats.raise_count / denom if denom > 0 else 0.0
            else:
                fold_rate, call_rate, raise_rate, aggression = 1/3, 1/3, 1/3, 0.0
        else:
            fold_rate, call_rate, raise_rate, aggression = 1/3, 1/3, 1/3, 0.0

        return torch.tensor([fold_rate, call_rate, raise_rate, aggression],
                            dtype=torch.float32)

    # ------------------------------------------------------------------
    # CFR infoset key construction (mirrors CFRAgent._obs_to_key)
    # ------------------------------------------------------------------

    @staticmethod
    def _build_cfr_key(hand: str, board: str, current_round: int,
                       action_history, viewer_id: int) -> str:
        """
        Build the CFR infoset key from the acting player's perspective.

        Delegates to belief_common.build_cfr_infoset_key().
        """
        return belief_common.build_cfr_infoset_key(
            hand, board, current_round, action_history
        )

    # ------------------------------------------------------------------
    # Nash + modulation likelihood computation
    # ------------------------------------------------------------------

    def get_nash_log_probs(self, hand: str, obs: Observation) -> torch.Tensor:
        """
        Get log-probabilities from the frozen CFR Nash strategy for the
        given hand in the current game state.

        Returns a (3,) tensor of log-probabilities over [FOLD, CALL, RAISE].
        """
        # Build infoset key from the perspective of the player holding `hand`
        board = obs.board if obs.board else ""
        action_history = obs.action_history if obs.action_history else []

        key = self._build_cfr_key(hand, board, obs.current_round,
                                  action_history, obs.current_player)

        # Get Nash strategy (numpy array of shape (3,))
        # Use all 3 actions as "legal" -- mask_and_normalize handles the rest
        all_actions = [Action.FOLD, Action.CALL, Action.RAISE]
        nash_probs = self.strategy_store.get_average_strategy(key, all_actions)

        # Clamp to avoid log(0)
        nash_probs = np.maximum(nash_probs, 1e-8)
        nash_probs = nash_probs / nash_probs.sum()  # re-normalize after clamp

        return torch.tensor(np.log(nash_probs), dtype=torch.float32)

    def get_adjusted_log_probs(self, hand: str, obs: Observation,
                               opp_stats: torch.Tensor = None) -> torch.Tensor:
        """
        Compute the modulated action log-probabilities:

            log P_adjusted = log_softmax( log pi_Nash + gate(stats) * delta(stats) )

        Args:
            hand: The hypothetical opponent hand card.
            obs: Current game observation.
            opp_stats: Pre-computed opponent stats tensor (4,). If None,
                       extracted from obs.

        Returns:
            (3,) tensor of log-probabilities over [FOLD, CALL, RAISE].
        """
        if opp_stats is None:
            opp_stats = self._encode_opp_stats(obs)

        # Nash base log-probs
        nash_log_probs = self.get_nash_log_probs(hand, obs)  # (3,)

        # Modulation
        stats_input = opp_stats.unsqueeze(0)  # (1, 4)
        gate = self.gate_net(stats_input).squeeze()  # scalar
        delta = self.delta_net(stats_input).squeeze()  # (3,)

        # Adjusted logits
        adjusted_logits = nash_log_probs + gate * delta  # (3,)

        # Return valid log-probabilities
        return torch.log_softmax(adjusted_logits, dim=-1)

    # ------------------------------------------------------------------
    # Belief computation
    # ------------------------------------------------------------------

    def initialize_belief(self, my_hand: str, board: str = None) -> np.ndarray:
        """
        Initialize P(opponent_hand) from card removal logic.
        """
        return belief_common.initialize_belief(my_hand, board)

    def update_belief(self, belief: np.ndarray, action: Action,
                      obs: Observation, opp_stats: torch.Tensor = None) -> np.ndarray:
        """
        Bayesian update of belief given an observed opponent action.

        Uses the modulated Nash likelihood:
            P(hand | action) proportional to P_adjusted(action | hand) * P(hand)
        """
        action_idx = int(action)
        likelihoods = np.zeros(3)

        if opp_stats is None:
            opp_stats = self._encode_opp_stats(obs)

        with torch.no_grad():
            for hand_idx in range(3):
                if belief[hand_idx] < 1e-8:
                    continue
                hand = self.CARDS[hand_idx]
                log_probs = self.get_adjusted_log_probs(hand, obs, opp_stats)
                likelihoods[hand_idx] = torch.exp(log_probs[action_idx]).item()

        posterior = belief * likelihoods
        total = posterior.sum()
        if total < 1e-10:
            return belief
        return posterior / total

    def compute_belief_from_history(self, obs: Observation) -> np.ndarray:
        """
        Compute the full belief vector by replaying the action history.

        Starts from the card-removal prior and applies Bayesian updates
        for each opponent action, using CFR Nash + modulation likelihoods.
        """
        opp_stats = self._encode_opp_stats(obs)

        def _update_callback(belief, action, running_state):
            # Convert running action_history list to tuple for Observation
            ah = running_state['action_history']
            action_history = tuple(ah) if ah else None

            hist_obs = Observation(
                player_hand=running_state['my_hand'],
                board=running_state['board'],
                pot=running_state['pot'],
                current_player=running_state['player_id'],
                current_round=running_state['current_round'],
                legal_actions=[],
                is_finished=False,
                raises_this_round=running_state['raises'],
                action_history=action_history,
            )
            return self.update_belief(belief, action, hist_obs, opp_stats)

        return belief_common.replay_belief_from_history(obs, _update_callback)

    @staticmethod
    def _build_action_history_up_to(full_history, target_player_id, target_action_name):
        """
        Build the action history tuple up to (but not including) the target action.

        This gives us the action history from the perspective of the game state
        at the moment the target player is about to act.
        """
        result = []
        for pid, aname in full_history:
            if pid == target_player_id and aname == target_action_name:
                break
            result.append((pid, aname))
        return tuple(result) if result else None

    # ------------------------------------------------------------------
    # Observation encoding
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

        hand_idx = self.CARD_MAP.get(obs.player_hand)
        hand_vec = torch.zeros(3)
        if hand_idx is not None:
            hand_vec[hand_idx] = 1.0

        board_idx = self.CARD_MAP.get(obs.board, 3)
        board_vec = torch.zeros(4)
        board_vec[board_idx] = 1.0

        p0_pot, p1_pot = obs.pot
        pot_rel = [p0_pot, p1_pot] if viewer_id == 0 else [p1_pot, p0_pot]
        pot_vec = torch.tensor(pot_rel, dtype=torch.float32) / self.MAX_CHIPS

        if belief is None:
            belief = self.compute_belief_from_history(obs)
        belief_vec = torch.tensor(belief, dtype=torch.float32)

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

        belief = self.compute_belief_from_history(obs)

        for action in obs.legal_actions:
            post_obs, done = LeducGame.simulate_action(obs, action)

            # Carry opponent_stats forward
            if obs.opponent_stats is not None:
                post_obs = replace(post_obs, opponent_stats=obs.opponent_stats)

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
                values = torch.tensor([r["value"] for r in results])
                probs = torch.softmax(values / self.temperature, dim=0)
                idx = torch.multinomial(probs, 1).item()
                return results[idx]["action"]
            else:
                return max(results, key=lambda x: x["value"])["action"]
        except Exception as e:
            print(f"Error in BeliefModulatedAgent selection: {e}")
            return results[0]["action"]
