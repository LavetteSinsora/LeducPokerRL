"""
Belief Stable Trainer — Self-play trainer with stable belief TD targets.

The key difference from BeliefTrainer:
  Standard belief TD(0):
    prediction = V(s_post_t, b_t)
    target = reward + gamma * V(s_post_{t+1}, b_{t+1})  <-- uses UPDATED belief

  Stable belief TD(0):
    prediction = V(s_post_t, b_t)
    target = reward + gamma * V(s_post_{t+1}, b_t)  <-- uses SAME belief as prediction

Why: When the opponent acts between t and t+1, our belief updates. This
belief change is noisy (depends on likelihood model quality). By using b_t
in the target, we prevent the model from learning to associate action values
with belief changes. The model learns only the intrinsic value of actions,
not the "value of information gained."

Trade-off: This breaks the TD chain. V(s_{t+1}, b_t) was never directly
trained as a prediction target for step t+1. If V depends heavily on belief,
the chain propagates values incorrectly.

Implementation: For each step in the chain, we record BOTH the encoding with
the current belief (for prediction) and the encoding with the current belief
applied to the next state (for the target). The standard chain used b_{t+1}
at state s_{t+1}; we override this by re-encoding s_{t+1} with b_t.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, List, Tuple
from src.engine.leduc_game import LeducGame, Action
from src.engine.observation import Observation
from src.agents.belief_stable import BeliefStableAgent
from src.training.base import BaseTrainer


class BeliefStableTrainer(BaseTrainer):
    """
    Trainer for BeliefStableAgent.

    Uses stable belief TD targets: the TD target at step t uses b_t
    (the belief at prediction time) instead of b_{t+1} (the updated belief
    after the opponent acts).
    """

    def __init__(self, agent: BeliefStableAgent, learning_rate: float = 1e-4,
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

    def collect_episode(self) -> Tuple[List[List[dict]], List[float], List[dict]]:
        """
        Play one episode of self-play.

        Unlike BeliefTrainer which stores just encoded tensors in chains,
        we store dicts with:
          - 'encoded': the encoded state with current belief (for prediction)
          - 'belief': the belief vector at this timestep (b_t)
          - 'post_obs': the post-action observation (for re-encoding with different beliefs)
          - 'viewer_id': the player this belongs to

        This allows us to re-encode s_{t+1} with b_t for the stable TD target.

        Returns:
            chains: per-player chain of step dicts
            rewards: final rewards for each player
            likelihood_data: data for training the likelihood model
        """
        self.game.reset()
        chains = [[], []]  # chains[p] = list of step dicts
        likelihood_data = []

        while not self.game.is_finished:
            current_player = self.game.current_player
            obs = self.game.get_observation(viewer_id=current_player)

            # Get action from belief agent
            action = self.agent.select_action(obs)
            if isinstance(action, tuple):
                action = action[0]

            # Compute belief at this point
            belief = self.agent.compute_belief_from_history(obs)

            # Record post-action state
            post_obs, _ = LeducGame.simulate_action(obs, action)
            encoded = self.agent.encode_observation(
                post_obs, viewer_id=current_player, belief=belief
            )

            chains[current_player].append({
                'encoded': encoded,
                'belief': belief.copy(),
                'post_obs': post_obs,
                'viewer_id': current_player,
            })

            # Record likelihood training data
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

            self.game.step(action)

        rewards = self.game.get_reward()
        return chains, rewards, likelihood_data

    def update_model(self, batch_data: list) -> float:
        """
        Update both the value network (stable TD) and likelihood model.

        Args:
            batch_data: List of (chains, rewards, likelihood_data) tuples.

        Returns:
            Combined loss value.
        """
        value_loss = self._update_value_network(batch_data)
        likelihood_loss = self._update_likelihood_model(batch_data)

        return value_loss + 0.1 * likelihood_loss

    def _update_value_network(self, batch_data: list) -> float:
        """
        Stable belief TD(0) update.

        For each step t:
          prediction = V(s_post_t, b_t)             -- uses step t's belief
          target = V(s_post_{t+1}, b_t) or reward    -- uses step t's belief, NOT b_{t+1}

        This is the critical difference from standard BeliefTrainer.
        """
        self.value_optimizer.zero_grad()
        total_losses = []

        for chains, rewards, _ in batch_data:
            for p_idx in [0, 1]:
                chain = chains[p_idx]
                if not chain:
                    continue

                for t in range(len(chain)):
                    # Prediction: V(s_post_t, b_t) — standard, same as regular trainer
                    prediction = self.agent.model(chain[t]['encoded']).squeeze(0)

                    if t == len(chain) - 1:
                        # Terminal: target is actual reward
                        target = torch.FloatTensor([rewards[p_idx]])
                    else:
                        # STABLE TARGET: re-encode s_{t+1} with b_t (current step's belief)
                        # instead of using chain[t+1]['encoded'] which has b_{t+1}
                        next_step = chain[t + 1]
                        stable_encoded = self.agent.encode_observation(
                            next_step['post_obs'],
                            viewer_id=next_step['viewer_id'],
                            belief=chain[t]['belief']  # <-- USE b_t, not b_{t+1}
                        )
                        with torch.no_grad():
                            target = self.agent.model(stable_encoded).squeeze(0)

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
        Identical to BeliefTrainer.
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
            "eval_type": "belief_stable",
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
