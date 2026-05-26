"""
Pruned History Agent — Adaptive value agent with pruned action history features.

Extends AdaptiveValueAgent with pruned intra-hand action history encoding that
removes permanently-zero fold count features. Since fold ends the hand immediately,
fold counts in action history are always zero — including them wastes model capacity.

Original AdaptiveHistoryAgent: 15 base + 4 stats + 16 history = 35 dims, 128 hidden
This agent (pruned):          15 base + 4 stats + 12 history = 31 dims, 64 hidden

Pruning per round (6 features instead of 8):
  - player_call_count (normalized)
  - player_raise_count (normalized)
  - opponent_call_count (normalized)
  - opponent_raise_count (normalized)
  - total_actions (normalized)
  - has_raise_flag (0 or 1)

Removed: player_fold_count, opponent_fold_count (always zero in action history)

For Leduc Hold'em (2 rounds): 6 * 2 = 12 extra features.
Total: 15 (base) + 4 (stats) + 12 (pruned history) = 31 features.

Uses 64 hidden units (same as AdaptiveValueAgent parent), avoiding the
parameter bloat of AdaptiveHistoryAgent's 128 hidden units.
"""

import torch
from dataclasses import replace
from src.engine.leduc_game import Action, LeducGame
from src.engine.observation import Observation
from .adaptive_value import AdaptiveValueAgent
from .value_based import ValueBasedAgent, ValueNetwork


# Action name strings used in game history tuples
_ACTION_TO_NAME = {
    Action.FOLD: "FOLD",
    Action.CALL: "CALL",
    Action.RAISE: "RAISE",
}


class PrunedHistoryAgent(AdaptiveValueAgent):
    """
    Adaptive value agent with pruned action history encoding.

    Combines opponent stats (4 features) with pruned action history (12 features).
    Pruning removes fold counts which are always zero in action history
    (fold ends the hand, so no subsequent actions are recorded).

    Total: 15 base + 4 stats + 12 history = 31 dims.
    Uses 64 hidden units (same as AdaptiveValueAgent parent).

    Inherits from AdaptiveValueAgent for opponent_stats handling.
    """

    HISTORY_FEATURES_PER_ROUND = 6  # call, raise per player + total + has_raise
    NUM_ROUNDS = 2
    PRUNED_HISTORY_SIZE = HISTORY_FEATURES_PER_ROUND * NUM_ROUNDS  # 12
    MAX_ACTIONS_PER_ROUND = 6

    def __init__(self, model_path: str = None, temperature: float = 1.0):
        # Total input: 15 base + 4 stats + 12 history = 31
        self.input_size = 15 + self.STATS_SIZE + self.PRUNED_HISTORY_SIZE  # 31
        self.temperature = temperature
        self.train_mode = False

        # 64 hidden units (same as parent AdaptiveValueAgent, NOT 128)
        self.model = ValueNetwork(self.input_size)
        if model_path:
            self.load_model(model_path)
        self.model.eval()

    def encode_observation(self, obs: Observation, viewer_id: int = None) -> torch.Tensor:
        """Encode base(15) + stats(4) + pruned_history(12) = 31 features."""
        if viewer_id is None:
            viewer_id = obs.current_player

        # Base features [1, 15] from ValueBasedAgent
        # (skip AdaptiveValueAgent.encode_observation which adds stats --
        #  we add stats and history manually in the correct order)
        base = ValueBasedAgent.encode_observation(self, obs, viewer_id)  # [1, 15]

        # Opponent stats [4]
        if obs.opponent_stats is not None and hasattr(obs.opponent_stats, 'to_feature_vector'):
            stats_vec = torch.tensor(obs.opponent_stats.to_feature_vector(),
                                     dtype=torch.float32)
        else:
            # No stats available -- uninformative default
            stats_vec = torch.tensor([0.5, 0.5, 0.5, 0.0], dtype=torch.float32)

        # Pruned action history [12]
        history_features = self._encode_pruned_action_history(obs.action_history, viewer_id)

        return torch.cat([base.squeeze(0), stats_vec, history_features]).unsqueeze(0)  # [1, 31]

    # -----------------------------------------------------------------
    # Pruned action history encoding
    # -----------------------------------------------------------------

    def _encode_pruned_action_history(self, action_history: tuple, viewer_id: int) -> torch.Tensor:
        """
        Encode action history into a fixed-size feature vector with fold counts removed.

        For each round, computes (6 features):
          - player_call_count / total_actions
          - player_raise_count / total_actions
          - opponent_call_count / total_actions
          - opponent_raise_count / total_actions
          - total_actions / MAX_ACTIONS_PER_ROUND
          - has_raise_flag (0 or 1)

        Fold counts are omitted because fold ends the hand immediately,
        so fold counts in action history are always zero.

        Args:
            action_history: Tuple of (player_id, action_name) pairs, or None.
            viewer_id: The player whose perspective we encode from.

        Returns:
            Tensor of shape (PRUNED_HISTORY_SIZE,) = (12,) for Leduc.
        """
        features = torch.zeros(self.PRUNED_HISTORY_SIZE)

        if not action_history:
            return features

        # Split actions into rounds
        round_actions = self._split_into_rounds(action_history)

        for round_idx, actions in enumerate(round_actions):
            if round_idx >= self.NUM_ROUNDS:
                break

            if not actions:
                continue

            total_actions = len(actions)
            offset = round_idx * self.HISTORY_FEATURES_PER_ROUND

            # Count actions per player (call and raise only, no fold)
            player_call = 0
            player_raise = 0
            opp_call = 0
            opp_raise = 0
            has_raise = False

            for (actor, action_name) in actions:
                if action_name == "RAISE":
                    has_raise = True

                if actor == viewer_id:
                    if action_name == "CALL":
                        player_call += 1
                    elif action_name == "RAISE":
                        player_raise += 1
                else:
                    if action_name == "CALL":
                        opp_call += 1
                    elif action_name == "RAISE":
                        opp_raise += 1

            # Normalize by total actions in the round
            norm = float(total_actions) if total_actions > 0 else 1.0

            features[offset + 0] = player_call / norm
            features[offset + 1] = player_raise / norm
            features[offset + 2] = opp_call / norm
            features[offset + 3] = opp_raise / norm

            # Summary features
            features[offset + 4] = total_actions / self.MAX_ACTIONS_PER_ROUND
            features[offset + 5] = 1.0 if has_raise else 0.0

        return features

    def _split_into_rounds(self, action_history: tuple) -> list:
        """
        Split a flat action history into per-round action lists.

        Uses Leduc game rules to determine round boundaries:
        - Round 0 ends when betting is complete (both players have acted and
          the last raise has been called, or both checked).
        - Remaining actions belong to round 1.

        Returns:
            List of lists, where each inner list contains (player, action_name) tuples
            for that round. Length <= NUM_ROUNDS.
        """
        if not action_history:
            return [[] for _ in range(self.NUM_ROUNDS)]

        rounds = [[] for _ in range(self.NUM_ROUNDS)]
        current_round = 0
        raises_this_round = 0
        actions_in_round = 0
        last_action_was_raise = False

        for (player, action_name) in action_history:
            if current_round >= self.NUM_ROUNDS:
                break

            rounds[current_round].append((player, action_name))
            actions_in_round += 1

            if action_name == "FOLD":
                # Game over, no more rounds
                break
            elif action_name == "RAISE":
                raises_this_round += 1
                last_action_was_raise = True
            elif action_name == "CALL":
                if last_action_was_raise:
                    # Calling a raise ends the round
                    current_round += 1
                    raises_this_round = 0
                    actions_in_round = 0
                    last_action_was_raise = False
                elif player == 1:
                    # P1 checks (calls with no raise) after P0 checked -> round ends
                    current_round += 1
                    raises_this_round = 0
                    actions_in_round = 0
                    last_action_was_raise = False

        return rounds

    # -----------------------------------------------------------------
    # 1-step lookahead carrying BOTH stats and history forward
    # -----------------------------------------------------------------

    def get_action_evaluations(self, obs: Observation) -> list:
        """1-step lookahead carrying BOTH opponent_stats AND action_history forward."""
        evaluations = []
        current_p = obs.current_player

        for action in obs.legal_actions:
            post_obs, done = LeducGame.simulate_action(obs, action)

            # Carry opponent_stats forward
            if obs.opponent_stats is not None:
                post_obs = replace(post_obs, opponent_stats=obs.opponent_stats)

            # Carry extended action_history forward
            action_name = _ACTION_TO_NAME[action]
            current_history = obs.action_history if obs.action_history else ()
            extended_history = current_history + ((current_p, action_name),)
            post_obs = replace(post_obs, action_history=extended_history)

            if done and action == Action.FOLD:
                val = -float(obs.pot[current_p])
            else:
                val = self._get_value(post_obs, viewer_id=current_p)

            encoded = self.encode_observation(post_obs, viewer_id=current_p)
            evaluations.append({
                "action": action,
                "value": val,
                "is_terminal": done,
                "encoded": encoded,
            })
        return evaluations
