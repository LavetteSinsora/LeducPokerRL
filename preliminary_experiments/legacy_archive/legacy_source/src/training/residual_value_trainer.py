"""
Residual Value Trainer — session-based training for the ResidualValueAgent.

Two modes:
  - Self-play (use_population=False): both players are the training agent.
  - Population (use_population=True): player 0 is the training agent, player 1
    is drawn from a rotating opponent pool (heuristic, value_based, adaptive_value).

Key difference from ModulatedValueTrainer: no gate in the value computation,
so delta_net gets the full gradient signal instead of gate-attenuated gradients.
"""

import os
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from dataclasses import replace
from typing import Dict, List, Tuple

from src.engine.leduc_game import LeducGame, Action
from src.engine.poker_session import PokerSession
from src.agents.base import BaseAgent
from src.training.adaptive_trainer import AdaptiveTrainer


class ResidualValueTrainer(AdaptiveTrainer):

    def __init__(self, agent: BaseAgent, learning_rate: float = 1e-4,
                 hands_per_session: int = 30, use_population: bool = False,
                 weight_decay: float = 1e-3, rotate_every: int = 100,
                 snapshot_every: int = 500):
        super().__init__(agent, learning_rate=learning_rate,
                         hands_per_session=hands_per_session)

        # Override optimizer: only delta_net, with L2 regularization
        self.optimizer = optim.Adam(
            agent.delta_net.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )

        self.use_population = use_population
        self.rotate_every = rotate_every
        self.snapshot_every = snapshot_every
        self.episode_count = 0

        if use_population:
            self.opponent_pool = self._build_initial_pool()
            self.current_opponent_idx = 0

    def _build_initial_pool(self):
        pool = []

        from src.agents.heuristic import HeuristicAgent
        pool.append(("heuristic", HeuristicAgent()))

        from src.agents.value_based import ValueBasedAgent
        vb = ValueBasedAgent()
        vb_path = "models/value_based_agent.pt"
        if os.path.exists(vb_path):
            vb.load_model(vb_path)
        vb.set_train_mode(False)
        pool.append(("value_based", vb))

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
        if self.episode_count > 0 and self.episode_count % self.rotate_every == 0:
            self.current_opponent_idx = (self.current_opponent_idx + 1) % len(self.opponent_pool)
            name = self.opponent_pool[self.current_opponent_idx][0]
            print(f"  Rotating opponent to: {name}")

    def _maybe_snapshot_self(self):
        if self.episode_count > 0 and self.episode_count % self.snapshot_every == 0:
            snapshot = copy.deepcopy(self.agent)
            snapshot.set_train_mode(False)
            name = f"self_snapshot_{self.episode_count}"
            self.opponent_pool.append((name, snapshot))
            print(f"  Added self-snapshot to pool (pool size: {len(self.opponent_pool)})")

    def collect_episode(self) -> List[Tuple[List[List], List[float]]]:
        """Play one session, storing (base_enc, stats_vec) tuples in chains."""
        self.session.reset()
        session_data = []
        opponent = self._get_current_opponent() if self.use_population else None

        for _ in range(self.hands_per_session):
            self.session.new_hand()
            chains = [[], []]

            while not self.session.is_finished:
                current_player = self.session.current_player
                obs = self.session.get_observation(viewer_id=current_player)

                if self.use_population and current_player == 1:
                    # Opponent from pool — no training data recorded
                    action = opponent.select_action(obs)
                    if isinstance(action, tuple):
                        action = action[0]
                else:
                    # Training agent (player 0 in population, both in self-play)
                    action = self.agent.select_action(obs)
                    if isinstance(action, tuple):
                        action = action[0]

                    post_obs, _ = LeducGame.simulate_action(obs, action)
                    if obs.opponent_stats is not None:
                        post_obs = replace(post_obs, opponent_stats=obs.opponent_stats)

                    base_enc = self.agent.encode_observation(post_obs, viewer_id=current_player)
                    stats_vec = self.agent._encode_stats(post_obs)
                    chains[current_player].append((base_enc, stats_vec))

                self.session.step(action)

            rewards = self.session.game.get_reward()
            session_data.append((chains, rewards))

        self.episode_count += self.hands_per_session
        if self.use_population:
            self._maybe_rotate_opponent()
            self._maybe_snapshot_self()

        return session_data

    def _compute_residual_value(self, base_enc, stats_vec):
        """Compute V_base + delta (no gate)."""
        with torch.no_grad():
            v_base = self.agent.model(base_enc)

        mod_input = torch.cat([base_enc.squeeze(0), stats_vec]).unsqueeze(0)
        delta = self.agent.delta_net(mod_input)

        return (v_base + delta).squeeze(0)

    def update_model(self, batch_data: list) -> float:
        """TD(0) update using residual value computation."""
        self.optimizer.zero_grad()
        total_losses = []

        for chains, rewards in batch_data:
            for p_idx in [0, 1]:
                chain = chains[p_idx]
                if not chain:
                    continue

                for t in range(len(chain)):
                    base_enc, stats_vec = chain[t]
                    prediction = self._compute_residual_value(base_enc, stats_vec)

                    if t == len(chain) - 1:
                        target = torch.FloatTensor([rewards[p_idx]])
                    else:
                        next_base_enc, next_stats_vec = chain[t + 1]
                        with torch.no_grad():
                            target = self._compute_residual_value(
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
        """Play one session and return a debug trace."""
        self.session.reset()
        episode_trace = []
        opponent = self._get_current_opponent() if self.use_population else None

        old_train_mode = self.agent.train_mode
        self.agent.set_train_mode(False)

        warmup_hands = min(5, self.hands_per_session - 1)

        for _ in range(warmup_hands):
            self.session.new_hand()
            while not self.session.is_finished:
                player = self.session.current_player
                obs = self.session.get_observation(viewer_id=player)
                if self.use_population and player == 1:
                    action = opponent.select_action(obs)
                else:
                    action = self.agent.select_action(obs)
                if isinstance(action, tuple):
                    action = action[0]
                self.session.step(action)

        self.session.new_hand()
        while not self.session.is_finished:
            current_player = self.session.current_player
            obs = self.session.get_observation(viewer_id=current_player)

            if self.use_population and current_player == 1:
                action = opponent.select_action(obs)
                if isinstance(action, tuple):
                    action = action[0]
            else:
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
                if self.use_population:
                    step_info["opponent_name"] = self._get_current_opponent_name()
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
        result = {
            "trace": episode_trace,
            "final_rewards": rewards,
            "eval_type": "value",
            "session_analytics": self.session.get_analytics(),
        }
        if self.use_population:
            result["opponent_name"] = self._get_current_opponent_name()
        return result

    def update_params(self, params: Dict):
        if "lr" in params:
            new_lr = params["lr"]
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = new_lr
            print(f"Learning rate updated to: {new_lr}")
        if "hands_per_session" in params:
            self.hands_per_session = params["hands_per_session"]
            print(f"Hands per session updated to: {self.hands_per_session}")
        if "rotate_every" in params and self.use_population:
            self.rotate_every = params["rotate_every"]
            print(f"Rotate every updated to: {self.rotate_every}")
        if "snapshot_every" in params and self.use_population:
            self.snapshot_every = params["snapshot_every"]
            print(f"Snapshot every updated to: {self.snapshot_every}")


class ResidualPopValueTrainer(ResidualValueTrainer):
    """Convenience subclass: ResidualValueTrainer with population mode enabled."""

    def __init__(self, agent: BaseAgent, learning_rate: float = 1e-4,
                 hands_per_session: int = 30):
        super().__init__(agent, learning_rate=learning_rate,
                         hands_per_session=hands_per_session,
                         use_population=True)
