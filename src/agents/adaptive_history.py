"""
Adaptive-History Combo Agent.

Combines AdaptiveValueAgent's opponent statistics (4 features) with
HistoryValueAgent's intra-hand action history encoding (16 features)
into a single wider-network agent.

Total observation: 15 base + 4 opponent stats + 16 action history = 35 dims.
Uses a wider network (128 hidden units, doubled from the default 64) to
handle the larger observation space.

Inherits from AdaptiveValueAgent for opponent_stats handling.
Copies HistoryValueAgent's encoding methods for action history.
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


class AdaptiveHistoryAgent(AdaptiveValueAgent):
    """
    Combines opponent stats (4 features) with action history encoding (16 features).

    Total: 15 base + 4 stats + 16 history = 35 dims.
    Uses a wider network (128 hidden) to handle the larger observation space.

    Inherits from AdaptiveValueAgent for opponent_stats handling.
    Uses HistoryValueAgent's encoding methods for action history.
    """

    FEATURES_PER_ROUND = 8
    NUM_ROUNDS = 2
    HISTORY_SIZE = FEATURES_PER_ROUND * NUM_ROUNDS  # 16
    MAX_ACTIONS_PER_ROUND = 6

    def __init__(self, model_path: str = None, temperature: float = 1.0):
        # Total input: 15 base + 4 stats + 16 history = 35
        self.input_size = 15 + self.STATS_SIZE + self.HISTORY_SIZE
        self.temperature = temperature
        self.train_mode = False

        # Wider network for larger input
        self.model = ValueNetwork(self.input_size, hidden_size=128)
        if model_path:
            self.load_model(model_path)
        self.model.eval()

    def encode_observation(self, obs: Observation, viewer_id: int = None) -> torch.Tensor:
        """Encode base(15) + stats(4) + history(16) = 35 features."""
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

        # Action history [16] -- reuse HistoryValueAgent's encoding logic
        history_features = self._encode_action_history(obs.action_history, viewer_id)

        return torch.cat([base.squeeze(0), stats_vec, history_features]).unsqueeze(0)  # [1, 35]

    # -----------------------------------------------------------------
    # Action history encoding (copied from HistoryValueAgent)
    # -----------------------------------------------------------------

    def _encode_action_history(self, action_history: tuple, viewer_id: int) -> torch.Tensor:
        """
        Encode action history into a fixed-size feature vector.

        For each round, computes:
          - Per-player action counts normalized by total actions in the round
          - Total actions normalized by max possible
          - Binary flag for whether any raise occurred

        Args:
            action_history: Tuple of (player_id, action_name) pairs, or None.
            viewer_id: The player whose perspective we encode from.
                       "player" = viewer_id, "opponent" = 1 - viewer_id.

        Returns:
            Tensor of shape (HISTORY_SIZE,) = (16,) for Leduc.
        """
        features = torch.zeros(self.HISTORY_SIZE)

        if not action_history:
            return features

        opponent_id = 1 - viewer_id

        # Split actions into rounds.
        round_actions = self._split_into_rounds(action_history)

        for round_idx, actions in enumerate(round_actions):
            if round_idx >= self.NUM_ROUNDS:
                break

            if not actions:
                continue

            total_actions = len(actions)
            offset = round_idx * self.FEATURES_PER_ROUND

            # Count actions per player per type
            player_counts = {"FOLD": 0, "CALL": 0, "RAISE": 0}
            opp_counts = {"FOLD": 0, "CALL": 0, "RAISE": 0}
            has_raise = False

            for (actor, action_name) in actions:
                if action_name == "RAISE":
                    has_raise = True

                if actor == viewer_id:
                    player_counts[action_name] = player_counts.get(action_name, 0) + 1
                else:
                    opp_counts[action_name] = opp_counts.get(action_name, 0) + 1

            # Normalize by total actions in the round
            norm = float(total_actions) if total_actions > 0 else 1.0

            features[offset + 0] = player_counts["FOLD"] / norm
            features[offset + 1] = player_counts["CALL"] / norm
            features[offset + 2] = player_counts["RAISE"] / norm
            features[offset + 3] = opp_counts["FOLD"] / norm
            features[offset + 4] = opp_counts["CALL"] / norm
            features[offset + 5] = opp_counts["RAISE"] / norm

            # Summary features
            features[offset + 6] = total_actions / self.MAX_ACTIONS_PER_ROUND
            features[offset + 7] = 1.0 if has_raise else 0.0

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
