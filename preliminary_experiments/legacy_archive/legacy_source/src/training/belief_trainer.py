"""
Belief Trainer — Self-play trainer for the Bayesian Belief Agent.

Training protocol:
  1. Self-play using the belief agent against itself.
  2. TD(0) on post-action state chains (same as SelfPlayTrainer).
  3. After each game, train the likelihood model using revealed opponent
     hands + their action sequences (both hands are visible in self-play).
  4. Uses PokerSession for multi-hand sessions with opponent stats.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, List, Tuple
from src.engine.leduc_game import LeducGame, Action
from src.engine.observation import Observation
from src.agents.belief_value import BeliefValueAgent
from src.training.base import BaseTrainer


class BeliefTrainer(BaseTrainer):
    """
    Trainer for BeliefValueAgent.

    Combines:
      - TD(0) value learning on post-action state chains
      - Cross-entropy likelihood model training on revealed opponent hands
    """

    def __init__(self, agent: BeliefValueAgent, learning_rate: float = 1e-4,
                 likelihood_lr: float = 5e-4):
        super().__init__(agent, eval_interval=50, eval_num_games=100)

        # Value network optimizer
        self.value_optimizer = optim.Adam(
            self.agent.model.parameters(), lr=learning_rate
        )
        self.value_criterion = nn.MSELoss()

        # Likelihood model optimizer
        self.likelihood_optimizer = optim.Adam(
            self.agent.likelihood_model.parameters(), lr=likelihood_lr
        )
        self.likelihood_criterion = nn.NLLLoss()

        self.game = LeducGame()

    def collect_episode(self) -> Tuple[List[List[torch.Tensor]], List[float],
                                       List[dict]]:
        """
        Play one episode of self-play.

        Returns:
            chains: per-player post-action state chains for TD(0)
            rewards: final rewards for each player
            likelihood_data: list of dicts with opponent hand + action info
                             for training the likelihood model
        """
        self.game.reset()
        chains = [[], []]  # chains[p] = P's post-action encoded states
        likelihood_data = []  # data for training the likelihood model

        # Track running state for belief computation
        action_history = []

        while not self.game.is_finished:
            current_player = self.game.current_player
            obs = self.game.get_observation(viewer_id=current_player)

            # Get action from belief agent
            action = self.agent.select_action(obs)
            if isinstance(action, tuple):
                action = action[0]

            # Compute belief at this point for encoding
            belief = self.agent.compute_belief_from_history(obs)

            # Record post-action state for TD(0)
            post_obs, _ = LeducGame.simulate_action(obs, action)
            encoded = self.agent.encode_observation(
                post_obs, viewer_id=current_player, belief=belief
            )
            chains[current_player].append(encoded)

            # Record likelihood training data:
            # We know both hands in self-play, so we record
            # (opponent_hand, action_taken, game_state_at_decision)
            # for training the likelihood model from the opponent's perspective
            opponent = 1 - current_player
            opponent_hand = self.game.player_hands[current_player]
            # From the opponent's perspective, the "opponent" is the current player
            # So we store the current player's hand as the hand being predicted
            likelihood_data.append({
                'actor': current_player,
                'actor_hand': self.game.player_hands[current_player],
                'action': action,
                'board': self.game.board,
                'pot': list(self.game.pot),
                'current_round': obs.current_round,
                'raises_this_round': obs.raises_this_round,
                'current_player': current_player,
            })

            action_history.append((current_player, action.name))
            self.game.step(action)

        rewards = self.game.get_reward()
        return chains, rewards, likelihood_data

    def update_model(self, batch_data: list) -> float:
        """
        Update both the value network (TD(0)) and likelihood model.

        Args:
            batch_data: List of (chains, rewards, likelihood_data) tuples.

        Returns:
            Combined loss value.
        """
        value_loss = self._update_value_network(batch_data)
        likelihood_loss = self._update_likelihood_model(batch_data)

        return value_loss + 0.1 * likelihood_loss  # Combined metric

    def _update_value_network(self, batch_data: list) -> float:
        """TD(0) update on post-action state chains."""
        self.value_optimizer.zero_grad()
        total_losses = []

        for chains, rewards, _ in batch_data:
            for p_idx in [0, 1]:
                chain = chains[p_idx]
                if not chain:
                    continue

                for t in range(len(chain)):
                    prediction = self.agent.model(chain[t]).squeeze(0)

                    if t == len(chain) - 1:
                        target = torch.FloatTensor([rewards[p_idx]])
                    else:
                        with torch.no_grad():
                            target = self.agent.model(chain[t + 1]).squeeze(0)

                    loss = self.value_criterion(prediction, target)
                    total_losses.append(loss)

        if total_losses:
            mean_loss = torch.stack(total_losses).mean()
            mean_loss.backward()
            self.value_optimizer.step()
            return mean_loss.item()
        return 0.0

    def _update_likelihood_model(self, batch_data: list) -> float:
        """
        Train the likelihood model P(action | hand, game_state).

        Uses revealed hands from self-play episodes.
        """
        self.likelihood_optimizer.zero_grad()
        total_losses = []

        for _, _, likelihood_data in batch_data:
            for entry in likelihood_data:
                hand = entry['actor_hand']
                action = entry['action']
                hand_idx = self.agent.CARD_MAP.get(hand)
                if hand_idx is None:
                    continue

                # Build observation for likelihood model input
                obs = Observation(
                    player_hand=hand,
                    board=entry['board'],
                    pot=entry['pot'],
                    current_player=entry['current_player'],
                    current_round=entry['current_round'],
                    legal_actions=[],
                    is_finished=False,
                    raises_this_round=entry['raises_this_round'],
                )

                inp = self.agent._encode_likelihood_input(hand_idx, obs)
                log_probs = self.agent.likelihood_model(inp)

                action_target = torch.LongTensor([int(action)])
                loss = self.likelihood_criterion(log_probs, action_target)
                total_losses.append(loss)

        if total_losses:
            mean_loss = torch.stack(total_losses).mean()
            mean_loss.backward()
            self.likelihood_optimizer.step()
            return mean_loss.item()
        return 0.0

    def debug_episode(self) -> Dict:
        """Play one episode and record detailed trace with belief vectors."""
        self.game.reset()
        episode_trace = []

        old_train_mode = self.agent.train_mode
        self.agent.set_train_mode(False)

        while not self.game.is_finished:
            current_player = self.game.current_player
            obs = self.game.get_observation(viewer_id=current_player)
            belief = self.agent.compute_belief_from_history(obs)

            evaluations = self.agent.get_action_evaluations(obs)
            selected_eval = max(evaluations, key=lambda x: x["value"])
            action = selected_eval["action"]

            step_info = {
                "player_id": current_player,
                "observation": {
                    "player_hand": obs.player_hand,
                    "board": obs.board,
                    "pot": obs.pot,
                    "current_round": obs.current_round,
                },
                "belief": belief.tolist(),
                "evaluations": [
                    {
                        "action": e["action"].name,
                        "value": e["value"],
                        "belief": e["belief"].tolist(),
                    } for e in evaluations
                ],
                "selected_action": action.name,
            }
            episode_trace.append(step_info)
            self.game.step(action)

        rewards = self.game.get_reward()

        for step in episode_trace:
            player_reward = rewards[step["player_id"]]
            step["true_value"] = player_reward

        self.agent.set_train_mode(old_train_mode)
        return {
            "trace": episode_trace,
            "final_rewards": rewards,
            "eval_type": "belief_value",
            "player_hands": list(self.game.player_hands),
        }

    def update_params(self, params: Dict):
        """Update learning rates."""
        if "lr" in params:
            for pg in self.value_optimizer.param_groups:
                pg['lr'] = params["lr"]
        if "likelihood_lr" in params:
            for pg in self.likelihood_optimizer.param_groups:
                pg['lr'] = params["likelihood_lr"]
