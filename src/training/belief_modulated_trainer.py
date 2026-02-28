"""
Belief-Modulated Trainer -- population-based training for the BeliefModulatedAgent.

Training protocol:
  1. Train against a rotating population of opponents (heuristic, value_based,
     adaptive_value) to expose the modulation layer to diverse play styles.
  2. TD(0) on post-action state chains for the value network.
  3. Cross-entropy loss on modulated likelihoods: when both hands are visible,
     compute NLLLoss(adjusted_log_probs, actual_action) to train the gate
     and delta networks.
  4. Uses PokerSession for multi-hand sessions with opponent stats tracking.

Key differences from BeliefTrainer:
  - NO learned likelihood MLP -- base likelihoods come from frozen CFR tables
  - Trains gate + delta networks via modulation_likelihood_loss
  - Population-based opponent rotation (like PopAdaptiveTrainer)
  - Combined loss: value_td_loss + 0.1 * modulation_likelihood_loss
"""

import os
import copy
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from dataclasses import replace
from typing import Dict, List, Tuple, Optional

from src.engine.leduc_game import LeducGame, Action
from src.engine.observation import Observation
from src.engine.poker_session import PokerSession
from src.agents.belief_modulated import BeliefModulatedAgent
from src.training.base import BaseTrainer


class BeliefModulatedTrainer(BaseTrainer):
    """
    Trainer for BeliefModulatedAgent.

    Combines:
      - TD(0) value learning on post-action state chains
      - NLL-based modulation loss on revealed opponent hands
      - Population-based opponent rotation for diverse training signals
    """

    def __init__(self, agent: BeliefModulatedAgent, learning_rate: float = 1e-4,
                 modulation_lr: float = 5e-4, hands_per_session: int = 30,
                 rotate_every: int = 100):
        super().__init__(agent, eval_interval=50, eval_num_games=100)

        self.hands_per_session = hands_per_session
        self.rotate_every = rotate_every
        self.session = PokerSession()

        # Value network optimizer
        self.value_optimizer = optim.Adam(
            self.agent.model.parameters(), lr=learning_rate
        )
        self.value_criterion = nn.MSELoss()

        # Modulation networks optimizer (gate + delta)
        self.modulation_optimizer = optim.Adam(
            list(self.agent.gate_net.parameters()) +
            list(self.agent.delta_net.parameters()),
            lr=modulation_lr,
        )
        self.modulation_criterion = nn.NLLLoss()

        # Build opponent pool
        self.opponent_pool = self._build_opponent_pool()
        self.current_opponent_idx = 0
        self.session_count = 0

    def _build_opponent_pool(self):
        """Create the opponent pool from pre-trained agents."""
        pool = []

        # 1. Heuristic -- rule-based baseline
        from src.agents.heuristic import HeuristicAgent
        pool.append(("heuristic", HeuristicAgent()))

        # 2. Value-based
        from src.agents.value_based import ValueBasedAgent
        vb = ValueBasedAgent()
        vb_path = "models/value_based_agent.pt"
        if os.path.exists(vb_path):
            vb.load_model(vb_path)
        vb.set_train_mode(False)
        pool.append(("value_based", vb))

        # 3. Adaptive value
        from src.agents.adaptive_value import AdaptiveValueAgent
        av = AdaptiveValueAgent()
        av_path = "models/adaptive_value_agent.pt"
        if os.path.exists(av_path):
            av.load_model(av_path)
        av.set_train_mode(False)
        pool.append(("adaptive_value", av))

        return pool

    def _get_current_opponent(self):
        return self.opponent_pool[self.current_opponent_idx][1]

    def _get_current_opponent_name(self):
        return self.opponent_pool[self.current_opponent_idx][0]

    def _maybe_rotate_opponent(self):
        if self.session_count > 0 and self.session_count % self.rotate_every == 0:
            self.current_opponent_idx = (
                (self.current_opponent_idx + 1) % len(self.opponent_pool)
            )
            name = self.opponent_pool[self.current_opponent_idx][0]
            print(f"  Rotating opponent to: {name}")

    def collect_episode(self) -> List[Tuple[List[List[torch.Tensor]], List[float], List[dict]]]:
        """
        Play one session of hands against a pool opponent.

        Returns a list of (chains, rewards, likelihood_data) tuples per hand.
          - chains: per-player post-action encoded states for TD(0)
          - rewards: terminal rewards for each player
          - likelihood_data: entries for training modulation networks
        """
        self.session.reset()
        session_data = []
        opponent = self._get_current_opponent()

        for _ in range(self.hands_per_session):
            self.session.new_hand()
            chains = [[], []]
            likelihood_data = []

            while not self.session.is_finished:
                current_player = self.session.current_player
                obs = self.session.get_observation(viewer_id=current_player)

                if current_player == 0:
                    # Training agent (player 0)
                    action = self.agent.select_action(obs)
                    if isinstance(action, tuple):
                        action = action[0]

                    # Record post-action state for TD(0)
                    belief = self.agent.compute_belief_from_history(obs)
                    post_obs, _ = LeducGame.simulate_action(obs, action)
                    if obs.opponent_stats is not None:
                        post_obs = replace(post_obs, opponent_stats=obs.opponent_stats)
                    encoded = self.agent.encode_observation(
                        post_obs, viewer_id=current_player, belief=belief
                    )
                    chains[current_player].append(encoded)

                    # Record likelihood data for player 0's action (visible to player 1)
                    # From player 1's perspective, player 0 is the opponent
                    likelihood_data.append({
                        'actor': current_player,
                        'actor_hand': self.session.game.player_hands[current_player],
                        'action': action,
                        'obs_at_decision': obs,
                    })
                else:
                    # Opponent from pool (player 1)
                    action = opponent.select_action(obs)
                    if isinstance(action, tuple):
                        action = action[0]

                    # Record likelihood data for opponent's action
                    # From player 0's perspective, player 1 is the opponent
                    likelihood_data.append({
                        'actor': current_player,
                        'actor_hand': self.session.game.player_hands[current_player],
                        'action': action,
                        'obs_at_decision': obs,
                    })

                self.session.step(action)

            rewards = self.session.game.get_reward()
            session_data.append((chains, rewards, likelihood_data))

        self.session_count += 1
        self._maybe_rotate_opponent()

        return session_data

    def update_model(self, batch_data: list) -> float:
        """
        Update both value network (TD(0)) and modulation networks (NLL).

        Args:
            batch_data: List of session data, where each session is a list
                        of (chains, rewards, likelihood_data) tuples.

        Returns:
            Combined loss: value_td_loss + 0.1 * modulation_likelihood_loss
        """
        # Flatten: batch_data is a list of session_data lists
        all_hands = []
        for session_data in batch_data:
            if isinstance(session_data, list):
                all_hands.extend(session_data)
            else:
                all_hands.append(session_data)

        value_loss = self._update_value_network(all_hands)
        mod_loss = self._update_modulation_networks(all_hands)

        return value_loss + 0.1 * mod_loss

    def _update_value_network(self, hand_data: list) -> float:
        """TD(0) update on post-action state chains (player 0 only)."""
        self.value_optimizer.zero_grad()
        total_losses = []

        for chains, rewards, _ in hand_data:
            # Only train on player 0's chain (our training agent)
            chain = chains[0]
            if not chain:
                continue

            for t in range(len(chain)):
                prediction = self.agent.model(chain[t]).squeeze(0)

                if t == len(chain) - 1:
                    target = torch.FloatTensor([rewards[0]])
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

    def _update_modulation_networks(self, hand_data: list) -> float:
        """
        Train gate and delta networks via NLL on modulated likelihoods.

        For each opponent decision where we know the true hand:
            loss = NLLLoss(adjusted_log_probs, actual_action)
        where adjusted_log_probs = log_softmax(log_pi_Nash + gate(stats) * delta(stats))
        """
        self.modulation_optimizer.zero_grad()
        total_losses = []

        for _, _, likelihood_data in hand_data:
            for entry in likelihood_data:
                # Only train modulation on opponent actions (player 1)
                # These are the actions where we want our likelihood model
                # to match the opponent's behavior
                if entry['actor'] != 1:
                    continue

                hand = entry['actor_hand']
                action = entry['action']
                obs = entry['obs_at_decision']

                if hand not in self.agent.CARD_MAP:
                    continue

                # Get opponent stats from the observation
                opp_stats = self.agent._encode_opp_stats(obs)

                # Compute adjusted log probs (with gradient through gate/delta)
                adjusted_log_probs = self.agent.get_adjusted_log_probs(
                    hand, obs, opp_stats
                )

                action_target = torch.LongTensor([int(action)])
                loss = self.modulation_criterion(
                    adjusted_log_probs.unsqueeze(0), action_target
                )
                total_losses.append(loss)

        if total_losses:
            mean_loss = torch.stack(total_losses).mean()
            mean_loss.backward()
            self.modulation_optimizer.step()
            return mean_loss.item()
        return 0.0

    def train(self, num_episodes: int, batch_size: int = 32, save_path: str = None,
              callback=None, start_episode: int = 0):
        """
        Session-based training loop with population rotation.

        Args:
            num_episodes: Number of sessions to run. Total hands =
                num_episodes * hands_per_session.
            batch_size: Number of individual hands to accumulate before updating.
            save_path: Where to persist the model on completion.
            callback: Progress callback.
            start_episode: Episode counter offset for resumed training.
        """
        self.agent.set_train_mode(True)
        self.stop_requested = False

        batch_data = []
        episode_counter = start_episode

        for session_idx in range(num_episodes):
            if self.stop_requested:
                print("Training stop requested.")
                break

            session_data = self.collect_episode()
            batch_data.extend(session_data)
            episode_counter += len(session_data)

            if len(batch_data) >= batch_size:
                loss = self.update_model(batch_data)
                batch_data = []

                if callback:
                    callback({
                        "episode": episode_counter,
                        "loss": loss,
                        "type": "batch_update",
                    })

                if session_idx < 2 or (episode_counter) % 100 == 0:
                    print(f"Session {session_idx + 1}, Episode {episode_counter}, "
                          f"Batch Loss: {loss:.4f}")

            # Evaluate periodically
            eval_every = max(1, self.eval_interval // self.hands_per_session)
            if (session_idx + 1) % eval_every == 0:
                avg_chips = self.evaluate(num_games=self.eval_num_games)
                if callback:
                    callback({
                        "episode": episode_counter,
                        "avg_chips_per_round": avg_chips,
                        "type": "evaluation",
                    })
                print(f"Session {session_idx + 1}, Episode {episode_counter}, "
                      f"Avg Chips/Round: {avg_chips:+.2f}")

        if save_path:
            os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
            self.agent.save_model(save_path)
            print(f"Model saved to {save_path}")

    def debug_episode(self) -> Dict:
        """Play one session against a pool opponent and return a debug trace."""
        self.session.reset()
        episode_trace = []
        opponent = self._get_current_opponent()

        old_train_mode = self.agent.train_mode
        self.agent.set_train_mode(False)

        warmup_hands = min(5, self.hands_per_session - 1)

        for _ in range(warmup_hands):
            self.session.new_hand()
            while not self.session.is_finished:
                player = self.session.current_player
                obs = self.session.get_observation(viewer_id=player)
                if player == 0:
                    action = self.agent.select_action(obs)
                else:
                    action = opponent.select_action(obs)
                if isinstance(action, tuple):
                    action = action[0]
                self.session.step(action)

        # Trace the final hand
        self.session.new_hand()
        while not self.session.is_finished:
            current_player = self.session.current_player
            obs = self.session.get_observation(viewer_id=current_player)

            if current_player == 0:
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
                    "opponent_name": self._get_current_opponent_name(),
                    "belief": self.agent.compute_belief_from_history(obs).tolist(),
                    "evaluations": [
                        {
                            "action": e["action"].name,
                            "value": e["value"],
                            "action_id": e["action"].value,
                            "belief": e["belief"].tolist(),
                        } for e in evaluations
                    ],
                    "selected_action": action.name,
                    "selected_action_id": action.value,
                }
                episode_trace.append(step_info)
            else:
                action = opponent.select_action(obs)
                if isinstance(action, tuple):
                    action = action[0]

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
            "eval_type": "belief_modulated",
            "session_analytics": self.session.get_analytics(),
            "opponent_name": self._get_current_opponent_name(),
        }

    def update_params(self, params: Dict):
        """Updates trainer parameters."""
        if "lr" in params:
            for pg in self.value_optimizer.param_groups:
                pg['lr'] = params["lr"]
        if "modulation_lr" in params:
            for pg in self.modulation_optimizer.param_groups:
                pg['lr'] = params["modulation_lr"]
        if "hands_per_session" in params:
            self.hands_per_session = params["hands_per_session"]
        if "rotate_every" in params:
            self.rotate_every = params["rotate_every"]
