"""
Opponent Response Planning Agent — 2-ply lookahead with learned opponent model.

This agent extends ValueBasedAgent with a learned opponent model that predicts
P(opponent_action | state). At decision time, instead of 1-ply lookahead
("what is the value of the post-state?"), it performs 2-ply lookahead:

    For each my_action in legal_actions:
        post_state = simulate(obs, my_action)
        if terminal: value = terminal_reward
        else:
            expected_value = 0
            opp_probs = opponent_model(encode(post_state, viewer=opponent))
            for opp_action in legal_actions(post_state):
                post_post = simulate(post_state, opp_action)
                if terminal: v = terminal_reward
                else: v = value_network(encode(post_post, viewer=me))
                expected_value += opp_probs[opp_action] * v
            value = expected_value
    Select my_action with highest value

This captures more strategic depth by asking "if I raise, the opponent will
probably call (70%) or reraise (30%), and accounting for their likely
response, the expected value is Y."
"""

import numpy as np
import torch
import torch.nn as nn
from src.engine.leduc_game import Action, LeducGame
from src.engine.observation import Observation
from .value_based import ValueBasedAgent, ValueNetwork
from .base import BaseAgent


class OpponentModel(nn.Module):
    """
    MLP classifier predicting P(opponent_action | state).
    Input: 15-dim state encoding (same as ValueBasedAgent)
    Output: 3 logits for [FOLD, CALL, RAISE]
    """

    def __init__(self, input_size: int = 15, hidden_size: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 3),  # 3 actions: FOLD, CALL, RAISE
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns raw logits (3-dim)."""
        return self.net(x)

    def predict_probs(self, x: torch.Tensor) -> torch.Tensor:
        """Returns action probabilities via softmax."""
        logits = self.forward(x)
        return torch.softmax(logits, dim=-1)


class OpponentModelAgent(ValueBasedAgent):
    """
    Agent with 2-ply lookahead using a learned opponent model.

    During action selection, it simulates each possible action, then
    uses the opponent model to predict how the opponent will respond,
    and evaluates the expected value across all opponent responses.

    The value network and opponent model both use the same 15-dim encoding
    as ValueBasedAgent. Training is via OpponentModelTrainer which trains
    both networks simultaneously through self-play.
    """

    def __init__(self, model_path: str = None, temperature: float = 1.0):
        # Initialize with same input_size as ValueBasedAgent (15)
        self.input_size = 15
        self.temperature = temperature
        self.train_mode = False

        self.model = ValueNetwork(self.input_size)
        self.opponent_model = OpponentModel(self.input_size)

        if model_path:
            self.load_model(model_path)
        self.model.eval()
        self.opponent_model.eval()

    def set_train_mode(self, mode: bool):
        self.train_mode = mode
        self.model.train(mode)
        self.opponent_model.train(mode)

    def save_model(self, path: str) -> None:
        """Save both value network and opponent model weights."""
        torch.save({
            'value_network': self.model.state_dict(),
            'opponent_model': self.opponent_model.state_dict(),
        }, path)

    def load_model(self, path: str) -> None:
        """Load both value network and opponent model weights."""
        checkpoint = torch.load(path, weights_only=False)
        if isinstance(checkpoint, dict) and 'value_network' in checkpoint:
            self.model.load_state_dict(checkpoint['value_network'])
            self.opponent_model.load_state_dict(checkpoint['opponent_model'])
        else:
            # Fallback: assume it's just a value network state dict
            self.model.load_state_dict(checkpoint)

    def _get_opponent_action_probs(self, obs: Observation, opponent_id: int) -> torch.Tensor:
        """
        Predict the opponent's action distribution for the given state.
        Returns a tensor of shape [3] with probabilities for [FOLD, CALL, RAISE].
        """
        encoded = self.encode_observation(obs, viewer_id=opponent_id)
        with torch.no_grad():
            probs = self.opponent_model.predict_probs(encoded)
        return probs.squeeze(0)  # [3]

    def get_action_evaluations_2ply(self, obs: Observation) -> list:
        """
        2-ply lookahead: for each action, simulate my action, then
        simulate the opponent's response weighted by the opponent model.
        """
        evaluations = []
        current_p = obs.current_player
        opponent_p = 1 - current_p

        for action in obs.legal_actions:
            # Ply 1: simulate my action
            post_obs, done = LeducGame.simulate_action(obs, action)

            if done:
                # Terminal after my action
                if action == Action.FOLD:
                    val = -float(obs.pot[current_p])
                else:
                    # Opponent folded or showdown triggered by my call
                    # Use the value network to estimate (showdown result unknown)
                    val = self._get_value(post_obs, viewer_id=current_p)
                evaluations.append({
                    "action": action,
                    "value": val,
                    "is_terminal": True,
                    "method": "terminal_1ply",
                })
                continue

            # Non-terminal: opponent must respond
            # post_obs.current_player should be the opponent
            opp_probs = self._get_opponent_action_probs(post_obs, opponent_p)

            # Legal actions available for opponent in the post-state
            opp_legal_actions = post_obs.legal_actions
            if not opp_legal_actions:
                # Fallback: if no legal actions, use value network
                val = self._get_value(post_obs, viewer_id=current_p)
                evaluations.append({
                    "action": action,
                    "value": val,
                    "is_terminal": False,
                    "method": "1ply_fallback",
                })
                continue

            # Ply 2: simulate each opponent response
            expected_value = 0.0
            total_prob = 0.0

            for opp_action in opp_legal_actions:
                opp_action_idx = opp_action.value  # FOLD=0, CALL=1, RAISE=2
                opp_prob = opp_probs[opp_action_idx].item()

                # Simulate opponent's action
                post_post_obs, done2 = LeducGame.simulate_action(post_obs, opp_action)

                if done2:
                    if opp_action == Action.FOLD:
                        # Opponent folds -> we win their pot contribution
                        v = float(post_obs.pot[opponent_p])
                    else:
                        # Terminal (showdown) after opponent's call
                        v = self._get_value(post_post_obs, viewer_id=current_p)
                else:
                    # Non-terminal: use value network to estimate from my perspective
                    v = self._get_value(post_post_obs, viewer_id=current_p)

                expected_value += opp_prob * v
                total_prob += opp_prob

            # Normalize if opponent model gave probability to illegal actions
            if total_prob > 0 and abs(total_prob - 1.0) > 1e-6:
                expected_value /= total_prob

            evaluations.append({
                "action": action,
                "value": expected_value,
                "is_terminal": False,
                "method": "2ply",
                "opp_probs": {a.name: opp_probs[a.value].item() for a in opp_legal_actions},
            })

        return evaluations

    def get_action_evaluations(self, obs: Observation) -> list:
        """Standard 1-ply evaluations (for comparison and training)."""
        return super().get_action_evaluations(obs)

    def select_action(self, obs: Observation) -> Action:
        """Select action using 2-ply lookahead."""
        results = self.get_action_evaluations_2ply(obs)

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
            print(f"Error in OpponentModelAgent selection: {e}")
            return results[0]["action"]

    def select_action_1ply(self, obs: Observation) -> Action:
        """Select action using standard 1-ply lookahead (for comparison)."""
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
        except Exception:
            return results[0]["action"]
