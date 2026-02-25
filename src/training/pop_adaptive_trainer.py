"""
Population-Diverse Adaptive Trainer.

Extends AdaptiveTrainer to train against a diverse opponent population
instead of pure self-play. The opponent pool includes rule-based, value-based,
and adaptive agents, with periodic snapshots of the training agent itself
added to the pool.

This forces the opponent_stats features to capture genuine behavioral
diversity, since different opponents have fundamentally different play
patterns.
"""

import os
import copy
import torch
from dataclasses import replace
from typing import Dict, List, Tuple

from src.engine.leduc_game import LeducGame, Action
from src.engine.poker_session import PokerSession
from src.agents.base import BaseAgent
from src.training.adaptive_trainer import AdaptiveTrainer


class PopAdaptiveTrainer(AdaptiveTrainer):
    """
    Trains PopAdaptiveAgent against a diverse opponent population.

    Instead of self-play, the agent faces a rotating pool of different
    opponent types. This forces genuine adaptation since different opponents
    have different play patterns for the opponent_stats features to capture.
    """

    def __init__(self, agent, learning_rate=1e-4, hands_per_session=30,
                 rotate_every=100, snapshot_every=500):
        super().__init__(agent, learning_rate=learning_rate,
                         hands_per_session=hands_per_session)
        self.rotate_every = rotate_every
        self.snapshot_every = snapshot_every
        self.opponent_pool = self._build_initial_pool()
        self.current_opponent_idx = 0
        self.episode_count = 0

    def _build_initial_pool(self):
        """Create the initial opponent pool from pre-trained agents."""
        pool = []

        # 1. Heuristic -- rule-based baseline
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

        # 3. Adaptive value -- try to load pre-trained model
        from src.agents.adaptive_value import AdaptiveValueAgent
        av = AdaptiveValueAgent()
        av_path = "models/adaptive_value_agent.pt"
        if os.path.exists(av_path):
            av.load_model(av_path)
        av.set_train_mode(False)
        pool.append(("adaptive_value", av))

        return pool

    def _get_current_opponent(self):
        """Get the current opponent from the pool."""
        return self.opponent_pool[self.current_opponent_idx][1]

    def _get_current_opponent_name(self):
        """Get the name of the current opponent."""
        return self.opponent_pool[self.current_opponent_idx][0]

    def _maybe_rotate_opponent(self):
        """Rotate to next opponent if it's time."""
        if self.episode_count > 0 and self.episode_count % self.rotate_every == 0:
            self.current_opponent_idx = (self.current_opponent_idx + 1) % len(self.opponent_pool)
            name = self.opponent_pool[self.current_opponent_idx][0]
            print(f"  Rotating opponent to: {name}")

    def _maybe_snapshot_self(self):
        """Add a frozen copy of the current agent to the pool."""
        if self.episode_count > 0 and self.episode_count % self.snapshot_every == 0:
            snapshot = copy.deepcopy(self.agent)
            snapshot.set_train_mode(False)
            name = f"self_snapshot_{self.episode_count}"
            self.opponent_pool.append((name, snapshot))
            print(f"  Added self-snapshot to pool (pool size: {len(self.opponent_pool)})")

    def collect_episode(self):
        """Play a session against the current pool opponent.

        Unlike the parent's self-play, one side is the training agent
        and the other is a fixed opponent from the pool.

        Returns a list of (chains, rewards) tuples -- one per hand in the session.
        """
        self.session.reset()
        session_data = []
        opponent = self._get_current_opponent()

        for _ in range(self.hands_per_session):
            self.session.new_hand()
            chains = [[], []]

            while not self.session.is_finished:
                current_player = self.session.current_player
                obs = self.session.get_observation(viewer_id=current_player)

                if current_player == 0:
                    # Training agent
                    action = self.agent.select_action(obs)
                    if isinstance(action, tuple):
                        action = action[0]

                    # Record post-action state for TD learning
                    post_obs, _ = LeducGame.simulate_action(obs, action)
                    if obs.opponent_stats is not None:
                        post_obs = replace(post_obs, opponent_stats=obs.opponent_stats)
                    encoded = self.agent.encode_observation(post_obs, viewer_id=current_player)
                    chains[current_player].append(encoded)
                else:
                    # Opponent from pool
                    action = opponent.select_action(obs)
                    if isinstance(action, tuple):
                        action = action[0]

                self.session.step(action)

            rewards = self.session.game.get_reward()
            session_data.append((chains, rewards))

        self.episode_count += self.hands_per_session
        self._maybe_rotate_opponent()
        self._maybe_snapshot_self()

        return session_data

    def debug_episode(self):
        """Play one session against a pool opponent and return a debug trace."""
        self.session.reset()
        episode_trace = []
        opponent = self._get_current_opponent()

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
                    "opponent_name": self._get_current_opponent_name(),
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
            "opponent_name": self._get_current_opponent_name(),
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
        if "rotate_every" in params:
            self.rotate_every = params["rotate_every"]
            print(f"Rotate every updated to: {self.rotate_every}")
        if "snapshot_every" in params:
            self.snapshot_every = params["snapshot_every"]
            print(f"Snapshot every updated to: {self.snapshot_every}")
