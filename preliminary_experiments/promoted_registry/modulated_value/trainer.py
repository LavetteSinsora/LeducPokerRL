"""
Modulated Value Trainer -- session-based training for the ModulatedValueAgent.

Extends AdaptiveTrainer since we need session-based training for opponent stats.
Key differences from AdaptiveTrainer:
  - Only optimizes mod_net and gate_net parameters (base is frozen)
  - Stores (base_enc, stats_vec) tuples in chains instead of single tensors
  - update_model computes modulated value through the three-network architecture
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from dataclasses import replace
from typing import Dict, List, Tuple

from agents.adaptive_value.trainer import AdaptiveTrainer
from agents.base import BaseAgent
from engine.leduc_game import LeducGame, Action
from engine.poker_session import PokerSession


class ModulatedValueTrainer(AdaptiveTrainer):

    def __init__(self, agent: BaseAgent, learning_rate: float = 1e-4,
                 hands_per_session: int = 30):
        # Call AdaptiveTrainer.__init__ which sets up session, etc.
        # But we need to override the optimizer to only train mod/gate
        super().__init__(agent, learning_rate=learning_rate,
                         hands_per_session=hands_per_session)

        # Override optimizer: only optimize modulation and gate networks
        self.optimizer = optim.Adam(
            list(agent.mod_net.parameters()) + list(agent.gate_net.parameters()),
            lr=learning_rate,
        )

    def collect_episode(self) -> List[Tuple[List[List], List[float]]]:
        """Play one session of hands, returning per-hand training data.

        Returns a list of (chains, rewards) tuples -- one per hand.
        Each chain entry stores a (base_enc, stats_vec) tuple instead
        of a single encoded tensor so update_model can compute the
        modulated value through separate networks.
        """
        self.session.reset()
        session_data = []

        for _ in range(self.hands_per_session):
            self.session.new_hand()
            chains = [[], []]

            while not self.session.is_finished:
                current_player = self.session.current_player
                obs = self.session.get_observation(viewer_id=current_player)

                action = self.agent.select_action(obs)
                if isinstance(action, tuple):
                    action = action[0]

                # Record post-action state with stats for TD learning
                post_obs, _ = LeducGame.simulate_action(obs, action)
                if obs.opponent_stats is not None:
                    post_obs = replace(post_obs, opponent_stats=obs.opponent_stats)

                # Store both base encoding and stats vector separately
                base_enc = self.agent.encode_observation(post_obs, viewer_id=current_player)
                stats_vec = self.agent._encode_stats(post_obs)
                chains[current_player].append((base_enc, stats_vec))

                self.session.step(action)

            rewards = self.session.game.get_reward()
            session_data.append((chains, rewards))

        return session_data

    def _compute_modulated_value(self, base_enc, stats_vec):
        """Compute V_base + gate * delta through the three networks.

        Args:
            base_enc: [1, 15] base encoding tensor
            stats_vec: [4] stats vector tensor

        Returns:
            Scalar modulated value tensor (with gradient for mod/gate)
        """
        # Base value (frozen, no grad flows through this)
        with torch.no_grad():
            v_base = self.agent.model(base_enc)  # [1, 1]

        # Modulation: concat base + stats
        mod_input = torch.cat([base_enc.squeeze(0), stats_vec]).unsqueeze(0)  # [1, 19]
        delta = self.agent.mod_net(mod_input)  # [1, 1]

        # Gate: stats only
        gate = self.agent.gate_net(stats_vec.unsqueeze(0))  # [1, 1]

        return (v_base + gate * delta).squeeze(0)  # [1]

    def update_model(self, batch_data: list) -> float:
        """TD(0) update using modulated value computation.

        Each chain entry is a (base_enc, stats_vec) tuple.
        """
        self.optimizer.zero_grad()
        total_losses = []

        for chains, rewards in batch_data:
            for p_idx in [0, 1]:
                chain = chains[p_idx]
                if not chain:
                    continue

                for t in range(len(chain)):
                    base_enc, stats_vec = chain[t]
                    prediction = self._compute_modulated_value(base_enc, stats_vec)

                    if t == len(chain) - 1:
                        # Last action -> target is terminal reward
                        target = torch.FloatTensor([rewards[p_idx]])
                    else:
                        # Bootstrap from next state
                        next_base_enc, next_stats_vec = chain[t + 1]
                        with torch.no_grad():
                            target = self._compute_modulated_value(
                                next_base_enc, next_stats_vec
                            ).detach()

                    loss = self.criterion(prediction, target)
                    total_losses.append(loss)

        if total_losses:
            mean_loss = torch.stack(total_losses).mean()
            mean_loss.backward()
            self.optimizer.step()
            return mean_loss.item()
        return 0.0

    def debug_episode(self) -> Dict:
        """Play one session and return a debug trace with stats visible."""
        self.session.reset()
        episode_trace = []

        old_train_mode = self.agent.train_mode
        self.agent.set_train_mode(False)

        # Play a short session to build up some stats, then trace the last hand
        warmup_hands = min(5, self.hands_per_session - 1)

        for _ in range(warmup_hands):
            self.session.new_hand()
            while not self.session.is_finished:
                player = self.session.current_player
                obs = self.session.get_observation(viewer_id=player)
                action = self.agent.select_action(obs)
                self.session.step(action)

        # Now trace the final hand with full detail
        self.session.new_hand()
        while not self.session.is_finished:
            current_player = self.session.current_player
            obs = self.session.get_observation(viewer_id=current_player)
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
                "opponent_stats": (obs.opponent_stats.to_feature_vector()
                                   if obs.opponent_stats else None),
                "evaluations": [
                    {
                        "action": e["action"].name,
                        "value": e["value"],
                        "action_id": e["action"].value,
                        "encoded_state": e["encoded"].squeeze(0).tolist(),
                    } for e in evaluations
                ],
                "selected_action": action.name,
                "selected_action_id": action.value,
                "encoded_state": selected_eval["encoded"].squeeze(0).tolist(),
            }
            episode_trace.append(step_info)
            self.session.step(action)

        rewards = self.session.game.get_reward()

        for step in episode_trace:
            player_reward = rewards[step["player_id"]]
            step["true_value"] = player_reward
            pred_val = next(e["value"] for e in step["evaluations"]
                           if e["action"] == step["selected_action"])
            step["prediction_error"] = (pred_val - player_reward) ** 2

        self.agent.set_train_mode(old_train_mode)
        return {
            "trace": episode_trace,
            "final_rewards": rewards,
            "eval_type": "value",
            "session_analytics": self.session.get_analytics(),
        }

    def update_params(self, params: Dict):
        """Updates trainer parameters."""
        if "lr" in params:
            new_lr = params["lr"]
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = new_lr
            print(f"Learning rate updated to: {new_lr}")
        if "hands_per_session" in params:
            self.hands_per_session = params["hands_per_session"]
            print(f"Hands per session updated to: {self.hands_per_session}")
