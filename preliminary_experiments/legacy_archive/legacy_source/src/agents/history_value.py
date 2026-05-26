"""
Action-History Value Agent.

Extends the ValueBasedAgent with a scalable intra-hand action history encoding.
The observation space is augmented with per-round action count summaries that
produce a fixed-size feature vector per round regardless of sequence length.

Encoding per round (8 features):
  - 6 normalized action counts (player fold/call/raise, opponent fold/call/raise)
  - 1 normalized total actions
  - 1 binary was-raise-made flag

For Leduc Hold'em (2 rounds): 8 * 2 = 16 extra features.
Total: 15 (base) + 16 (history) = 31 features.

This design scales to Texas Hold'em (4 rounds) by simply increasing NUM_ROUNDS.
"""

import torch
from dataclasses import replace
from src.engine.leduc_game import Action, LeducGame
from src.engine.observation import Observation
from .value_based import ValueBasedAgent, ValueNetwork


# Action name strings used in game history tuples
_ACTION_NAMES = ("FOLD", "CALL", "RAISE")

# Mapping from Action enum to the name string used in history
_ACTION_TO_NAME = {
    Action.FOLD: "FOLD",
    Action.CALL: "CALL",
    Action.RAISE: "RAISE",
}


class HistoryValueAgent(ValueBasedAgent):
    """
    Value-based agent augmented with intra-hand action history features.

    The action history encoding uses per-round action count summaries,
    which scale to any number of betting rounds (Leduc: 2, Texas Hold'em: 4)
    without changing the encoding logic -- only NUM_ROUNDS needs updating.
    """

    FEATURES_PER_ROUND = 8  # 6 action counts + 2 summaries
    NUM_ROUNDS = 2          # Leduc has 2 rounds; set to 4 for Texas Hold'em
    HISTORY_SIZE = FEATURES_PER_ROUND * NUM_ROUNDS  # 16 for Leduc

    # Maximum possible actions per round (used for normalization).
    # In Leduc: max 2 raises, each raise needs a response, plus initial actions.
    # Conservatively set to 6 to keep normalization reasonable.
    MAX_ACTIONS_PER_ROUND = 6

    def __init__(self, model_path: str = None, temperature: float = 1.0):
        # Bypass ValueBasedAgent.__init__ to set correct input_size
        self.input_size = 15 + self.HISTORY_SIZE  # 31 for Leduc
        self.temperature = temperature
        self.train_mode = False

        self.model = ValueNetwork(self.input_size)
        if model_path:
            self.load_model(model_path)
        self.model.eval()

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
        # In Leduc, round 0 actions come before the board card is dealt (round transition).
        # We detect round boundaries by tracking when the round changes.
        # Since action_history is a flat sequence, we infer rounds from the game structure:
        # all actions before the first round transition belong to round 0, rest to round 1.
        #
        # Heuristic: In Leduc, the transition from round 0 to round 1 happens when betting
        # ends in round 0. We don't have explicit round markers in the history, so we
        # reconstruct round boundaries by replaying the action sequence.
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
        # Track whether the round's betting is complete
        # In Leduc: P0 acts first. Betting ends when:
        #   - P1 calls/checks after P0 checks (both check)
        #   - A raise is called (caller doesn't re-raise)
        #   - A fold occurs (game ends)
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
                    # P0 always acts first in a round, so if P1 calls and no raise, round over
                    current_round += 1
                    raises_this_round = 0
                    actions_in_round = 0
                    last_action_was_raise = False

        return rounds

    def encode_observation(self, obs: Observation, viewer_id: int = None) -> torch.Tensor:
        """
        Encodes base features (15) + action history features (16) = 31 features.
        """
        if viewer_id is None:
            viewer_id = obs.current_player

        base = super().encode_observation(obs, viewer_id)  # [1, 15]

        history_features = self._encode_action_history(
            obs.action_history, viewer_id
        )  # (16,)

        return torch.cat([base.squeeze(0), history_features]).unsqueeze(0)  # [1, 31]

    def get_action_evaluations(self, obs: Observation) -> list:
        """
        1-step lookahead, extending action_history for simulated post-states.

        LeducGame.simulate_action() does not carry action_history forward,
        so we manually append the taken action to create the correct history
        for each simulated successor state.
        """
        evaluations = []
        current_p = obs.current_player

        for action in obs.legal_actions:
            post_obs, done = LeducGame.simulate_action(obs, action)

            # Build the extended action_history for this simulated state
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
