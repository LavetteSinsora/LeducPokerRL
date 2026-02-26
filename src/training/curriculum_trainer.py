"""
Curriculum-Based Population Trainer.

Extends AdaptiveTrainer with three key improvements over PopAdaptiveTrainer:

1. Block scheduling: trains against one opponent for block_size sessions before
   rotating, allowing the agent to focus and the stats to accumulate properly.

2. Rehearsal buffer: stores (chains, rewards) from past opponents and mixes
   them into each update batch to prevent catastrophic forgetting.

3. Both-player training: records TD chains for both player 0 (training agent)
   and player 1 (opponent position), giving the value network training signal
   from both sides of the table.

These fixes address the three failure modes of PopAdaptiveAgent:
  - Only 67.6% training data (only player 0 chain was trained)
  - Conflicting gradients from diverse opponents
  - Random rotation disrupted stat accumulation
"""

import os
import random
import torch
from dataclasses import replace
from typing import Dict, List, Tuple

from src.engine.leduc_game import LeducGame, Action
from src.engine.poker_session import PokerSession
from src.agents.base import BaseAgent
from src.training.adaptive_trainer import AdaptiveTrainer


class CurriculumTrainer(AdaptiveTrainer):
    """
    Trains CurriculumAgent against a sequenced opponent population
    with block scheduling and experience rehearsal.
    """

    def __init__(self, agent, learning_rate=1e-4, hands_per_session=30,
                 block_size=100, rehearsal_ratio=0.2, max_buffer_size=5000):
        super().__init__(agent, learning_rate=learning_rate,
                         hands_per_session=hands_per_session)
        self.block_size = block_size  # sessions per opponent block
        self.rehearsal_ratio = rehearsal_ratio  # fraction of batch from old data
        self.max_buffer_size = max_buffer_size
        self.rehearsal_buffer = []  # stores (chains, rewards) from past opponents
        self.opponent_pool = self._build_opponent_pool()
        self.current_opponent_idx = 0
        self.sessions_in_current_block = 0
        self.forgetting_log = []  # track per-opponent performance over time

    def _build_opponent_pool(self):
        """Create opponent pool ordered weak to strong."""
        pool = []

        # 1. Heuristic -- rule-based baseline (weakest)
        from src.agents.heuristic import HeuristicAgent
        pool.append(("heuristic", HeuristicAgent()))

        # 2. Value-based -- try to load pre-trained model
        from src.agents.value_based import ValueBasedAgent
        vb = ValueBasedAgent()
        vb_path = "models/value_based_agent.pt"
        if os.path.exists(vb_path):
            vb.load_model(vb_path)
        vb.set_train_mode(False)
        pool.append(("value_based", vb))

        # 3. Adaptive value -- try to load pre-trained model (strongest)
        from src.agents.adaptive_value import AdaptiveValueAgent
        av = AdaptiveValueAgent()
        av_path = "models/adaptive_value_agent.pt"
        if os.path.exists(av_path):
            av.load_model(av_path)
        av.set_train_mode(False)
        pool.append(("adaptive_value", av))

        return pool

    def _transition_to_next_opponent(self):
        """Advance to the next opponent in the curriculum."""
        self.sessions_in_current_block = 0
        self.current_opponent_idx = (self.current_opponent_idx + 1) % len(self.opponent_pool)
        name = self.opponent_pool[self.current_opponent_idx][0]
        print(f"  Block transition: now training against {name}")

    def collect_episode(self):
        """Play a session against the current block opponent.

        Unlike PopAdaptiveTrainer, records chains for BOTH players
        (player 0 = training agent, player 1 = opponent). The training
        agent's value network estimates values for both player positions.

        Returns a list of (chains, rewards) tuples -- one per hand.
        """
        self.session.reset()
        session_data = []
        opponent = self.opponent_pool[self.current_opponent_idx][1]

        for _ in range(self.hands_per_session):
            self.session.new_hand()
            chains = [[], []]

            while not self.session.is_finished:
                current_player = self.session.current_player
                obs = self.session.get_observation(viewer_id=current_player)

                if current_player == 0:
                    action = self.agent.select_action(obs)
                else:
                    action = opponent.select_action(obs)
                if isinstance(action, tuple):
                    action = action[0]

                # Record post-action state for BOTH players
                post_obs, _ = LeducGame.simulate_action(obs, action)
                if obs.opponent_stats is not None:
                    post_obs = replace(post_obs, opponent_stats=obs.opponent_stats)
                encoded = self.agent.encode_observation(post_obs, viewer_id=current_player)
                chains[current_player].append(encoded)

                self.session.step(action)

            rewards = self.session.game.get_reward()
            session_data.append((chains, rewards))

        self.sessions_in_current_block += 1

        # Store in rehearsal buffer
        for item in session_data:
            if len(self.rehearsal_buffer) < self.max_buffer_size:
                self.rehearsal_buffer.append(item)
            else:
                # Random replacement
                idx = random.randint(0, len(self.rehearsal_buffer) - 1)
                self.rehearsal_buffer[idx] = item

        # Block transition
        if self.sessions_in_current_block >= self.block_size:
            self._transition_to_next_opponent()

        return session_data

    def update_model(self, batch_data):
        """TD(0) update with rehearsal mixing.

        Mixes in a fraction of rehearsal_ratio data from the rehearsal buffer
        to prevent catastrophic forgetting of earlier opponents.
        """
        # Mix in rehearsal data
        if self.rehearsal_buffer and self.rehearsal_ratio > 0:
            n_rehearsal = max(1, int(len(batch_data) * self.rehearsal_ratio
                                     / (1 - self.rehearsal_ratio)))
            n_rehearsal = min(n_rehearsal, len(self.rehearsal_buffer))
            rehearsal_samples = random.sample(self.rehearsal_buffer, n_rehearsal)
            batch_data = batch_data + rehearsal_samples

        return super().update_model(batch_data)

    def evaluate_against_all_opponents(self, num_rounds=100):
        """Evaluate current agent against all opponents for forgetting detection."""
        from src.training.evaluation import quick_evaluate
        results = {}
        old_mode = self.agent.train_mode
        self.agent.set_train_mode(False)
        for name, opponent in self.opponent_pool:
            score = quick_evaluate(self.agent, opponent, num_rounds=num_rounds)
            results[name] = score
        self.agent.set_train_mode(old_mode)
        self.forgetting_log.append({
            "sessions_trained": self.sessions_in_current_block
                                  + self.current_opponent_idx * self.block_size,
            "current_opponent": self.opponent_pool[self.current_opponent_idx][0],
            "scores": results,
        })
        return results

    def debug_episode(self):
        """Play one session against a pool opponent and return a debug trace."""
        self.session.reset()
        episode_trace = []
        opponent = self.opponent_pool[self.current_opponent_idx][1]

        old_train_mode = self.agent.train_mode
        self.agent.set_train_mode(False)

        # Play warmup hands to build up stats
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

        # Trace the final hand with full detail
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
                    "opponent_name": self.opponent_pool[self.current_opponent_idx][0],
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
            "eval_type": "value",
            "session_analytics": self.session.get_analytics(),
            "opponent_name": self.opponent_pool[self.current_opponent_idx][0],
        }

    def update_params(self, params: Dict):
        """Updates trainer parameters."""
        if "lr" in params:
            new_lr = params["lr"]
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = new_lr
            print(f"Learning rate updated to: {new_lr}")
        if "hands_per_session" in params:
            self.hands_per_session = params["hands_per_session"]
            print(f"Hands per session updated to: {self.hands_per_session}")
        if "block_size" in params:
            self.block_size = params["block_size"]
            print(f"Block size updated to: {self.block_size}")
        if "rehearsal_ratio" in params:
            self.rehearsal_ratio = params["rehearsal_ratio"]
            print(f"Rehearsal ratio updated to: {self.rehearsal_ratio}")
