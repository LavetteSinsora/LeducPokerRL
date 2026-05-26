"""Trainer for opponent-encoder modulation with a population opponent pool."""

import copy
import os
from dataclasses import replace
from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.optim as optim

from preliminary_experiments.promoted_registry.base_trainer import BaseTrainer
from agents.registry import registry
from engine.leduc_game import LeducGame
from engine.poker_session import PokerSession


class OpponentEncoderModulationTrainer(BaseTrainer):
    """Train only the experimental head while keeping the base value net frozen."""

    def __init__(
        self,
        agent,
        learning_rate: float = 1e-4,
        hands_per_session: int = 30,
        action_loss_weight: float = 0.5,
        gate_target: float = 0.4,
        gate_reg_weight: float = 0.5,
        modulation_reg_weight: float = 0.5,
        rotate_every: int = 200,
        snapshot_every: int = 2000,
        opponent_ids: Sequence[str] = ("heuristic", "value_based", "adaptive_value", "modulated_value", "cfr"),
    ):
        super().__init__(agent, eval_interval=200, eval_num_games=200)
        self.hands_per_session = hands_per_session
        self.action_loss_weight = action_loss_weight
        self.gate_target = gate_target
        self.gate_reg_weight = gate_reg_weight
        self.modulation_reg_weight = modulation_reg_weight
        self.rotate_every = rotate_every
        self.snapshot_every = snapshot_every
        self.session = PokerSession()

        self.optimizer = optim.Adam(
            list(agent.encoder.parameters())
            + list(agent.mod_net.parameters())
            + list(agent.gate_net.parameters())
            + list(agent.action_head.parameters()),
            lr=learning_rate,
        )
        self.value_criterion = nn.MSELoss()
        self.action_criterion = nn.CrossEntropyLoss()

        self.opponent_pool = self._build_opponent_pool(opponent_ids)
        self.current_opponent_idx = 0
        self.sessions_played = 0
        self.last_metrics = {
            "loss": 0.0,
            "value_loss": 0.0,
            "action_loss": 0.0,
            "gate_reg_loss": 0.0,
            "modulation_reg_loss": 0.0,
        }

    def _build_opponent_pool(self, opponent_ids: Sequence[str]) -> List[Tuple[str, object]]:
        pool: List[Tuple[str, object]] = []
        for agent_id in opponent_ids:
            opponent = registry.create(agent_id)
            checkpoint_path = registry.get_checkpoint_path(agent_id)
            if checkpoint_path and os.path.exists(checkpoint_path):
                try:
                    opponent.load_model(checkpoint_path)
                except Exception as exc:
                    print(f"Warning: could not load {agent_id} checkpoint: {exc}")
            opponent.set_train_mode(False)
            pool.append((agent_id, opponent))

        if not pool:
            raise ValueError("Opponent pool is empty")
        return pool

    def _get_current_opponent(self):
        return self.opponent_pool[self.current_opponent_idx][1]

    def _get_current_opponent_name(self) -> str:
        return self.opponent_pool[self.current_opponent_idx][0]

    def _maybe_rotate_opponent(self):
        if self.sessions_played > 0 and self.sessions_played % self.rotate_every == 0:
            self.current_opponent_idx = (self.current_opponent_idx + 1) % len(self.opponent_pool)
            print(f"  Rotating opponent to: {self._get_current_opponent_name()}")

    def _maybe_snapshot_self(self):
        if self.sessions_played > 0 and self.sessions_played % self.snapshot_every == 0:
            snapshot = copy.deepcopy(self.agent)
            snapshot.set_train_mode(False)
            name = f"self_snapshot_{self.sessions_played}"
            self.opponent_pool.append((name, snapshot))
            print(f"  Added self snapshot to opponent pool ({name})")

    def collect_episode(self):
        """Play one session against the current opponent."""
        self.session.reset()
        session_data = []
        opponent = self._get_current_opponent()

        for _ in range(self.hands_per_session):
            self.session.new_hand()
            value_chain = []
            action_examples = []

            while not self.session.is_finished:
                current_player = self.session.current_player

                if current_player == 0:
                    obs = self.session.get_observation(viewer_id=0)
                    action = self.agent.select_action(obs)
                    if isinstance(action, tuple):
                        action = action[0]

                    post_obs, _ = LeducGame.simulate_action(obs, action)
                    if obs.opponent_stats is not None:
                        post_obs = replace(post_obs, opponent_stats=obs.opponent_stats)

                    state_encoding = self.agent.encode_observation(post_obs, viewer_id=0)
                    stats_vec = self.agent.encode_macro_stats(post_obs)
                    value_chain.append((state_encoding, stats_vec))
                else:
                    opponent_obs = self.session.get_observation(viewer_id=1)
                    agent_view_obs = self.session.get_observation(viewer_id=0)
                    action = opponent.select_action(opponent_obs)
                    if isinstance(action, tuple):
                        action = action[0]

                    state_encoding = self.agent.encode_observation(agent_view_obs, viewer_id=0)
                    stats_vec = self.agent.encode_macro_stats(agent_view_obs)
                    action_examples.append((state_encoding, stats_vec, action.value))

                self.session.step(action)

            rewards = self.session.game.get_reward()
            session_data.append(
                {
                    "value_chain": value_chain,
                    "reward": rewards[0],
                    "action_examples": action_examples,
                }
            )

        self.sessions_played += 1
        self._maybe_rotate_opponent()
        self._maybe_snapshot_self()
        return session_data

    def update_model(self, batch_data: list) -> float:
        self.optimizer.zero_grad()

        value_losses = []
        gate_reg_losses = []
        modulation_reg_losses = []
        action_state_batch = []
        action_stats_batch = []
        action_targets = []

        for sample in batch_data:
            chain = sample["value_chain"]
            reward = sample["reward"]

            for t, (state_encoding, stats_vec) in enumerate(chain):
                parts = self.agent.predict_value_from_encoded(
                    state_encoding,
                    stats_vec,
                    return_parts=True,
                )
                prediction = parts["value"].squeeze(0)
                gate_reg_losses.append((parts["gate"] - self.gate_target).pow(2).mean())
                modulation_reg_losses.append((parts["gate"] * parts["delta"]).pow(2).mean())

                if t == len(chain) - 1:
                    target = torch.tensor([reward], dtype=torch.float32)
                else:
                    next_state, next_stats = chain[t + 1]
                    with torch.no_grad():
                        target = self.agent.predict_value_from_encoded(
                            next_state,
                            next_stats,
                        ).detach().squeeze(0)

                value_losses.append(self.value_criterion(prediction, target))

            for state_encoding, stats_vec, action_idx in sample["action_examples"]:
                action_state_batch.append(state_encoding.squeeze(0))
                action_stats_batch.append(stats_vec)
                action_targets.append(action_idx)

        total_loss = torch.tensor(0.0)
        value_loss_val = 0.0
        action_loss_val = 0.0
        gate_reg_val = 0.0
        modulation_reg_val = 0.0

        if value_losses:
            mean_value_loss = torch.stack(value_losses).mean()
            total_loss = total_loss + mean_value_loss
            value_loss_val = mean_value_loss.item()

        if action_state_batch:
            states = torch.stack(action_state_batch)
            stats = torch.stack(action_stats_batch)
            targets = torch.tensor(action_targets, dtype=torch.long)
            logits = self.agent.predict_action_logits(states, stats)
            action_loss = self.action_criterion(logits, targets)
            total_loss = total_loss + self.action_loss_weight * action_loss
            action_loss_val = action_loss.item()

        if gate_reg_losses:
            gate_reg = torch.stack(gate_reg_losses).mean()
            total_loss = total_loss + self.gate_reg_weight * gate_reg
            gate_reg_val = gate_reg.item()

        if modulation_reg_losses:
            modulation_reg = torch.stack(modulation_reg_losses).mean()
            total_loss = total_loss + self.modulation_reg_weight * modulation_reg
            modulation_reg_val = modulation_reg.item()

        if total_loss.requires_grad:
            total_loss.backward()
            self.optimizer.step()

        self.last_metrics = {
            "loss": total_loss.item() if total_loss.requires_grad else 0.0,
            "value_loss": value_loss_val,
            "action_loss": action_loss_val,
            "gate_reg_loss": gate_reg_val,
            "modulation_reg_loss": modulation_reg_val,
        }
        return self.last_metrics["loss"]

    def train(
        self,
        num_episodes: int,
        batch_size: int = 32,
        save_path: str = None,
        callback=None,
        start_episode: int = 0,
    ):
        self.agent.set_train_mode(True)
        self.stop_requested = False
        batch_data = []

        for session_idx in range(num_episodes):
            if self.stop_requested:
                print("Training stop requested.")
                break

            session_data = self.collect_episode()
            batch_data.extend(session_data)
            session_number = start_episode + session_idx + 1
            hands_seen = session_number * self.hands_per_session

            if len(batch_data) >= batch_size:
                self.update_model(batch_data)
                batch_data = []

                if callback:
                    callback(
                        {
                            "episode": session_number,
                            "hands_seen": hands_seen,
                            "opponent": self._get_current_opponent_name(),
                            "type": "batch_update",
                            **self.last_metrics,
                        }
                    )

                if session_idx < 2 or session_number % 50 == 0:
                    print(
                        f"Session {session_number}, Hands {hands_seen}, "
                        f"Loss {self.last_metrics['loss']:.4f}, "
                        f"Value {self.last_metrics['value_loss']:.4f}, "
                        f"Action {self.last_metrics['action_loss']:.4f}, "
                        f"GateReg {self.last_metrics['gate_reg_loss']:.4f}, "
                        f"ModReg {self.last_metrics['modulation_reg_loss']:.4f}"
                    )

            if session_number % self.eval_interval == 0:
                avg_chips = self.evaluate(num_games=self.eval_num_games)
                if callback:
                    callback(
                        {
                            "episode": session_number,
                            "hands_seen": hands_seen,
                            "avg_chips_per_round": avg_chips,
                            "opponent": self._get_current_opponent_name(),
                            "type": "evaluation",
                        }
                    )
                print(
                    f"Session {session_number}, Hands {hands_seen}, "
                    f"Avg Chips/Round {avg_chips:+.3f}"
                )

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            self.agent.save_model(save_path)
            print(f"Model saved to {save_path}")

    def debug_episode(self) -> Dict:
        self.session.reset()
        episode_trace = []
        opponent = self._get_current_opponent()

        old_train_mode = self.agent.train_mode
        self.agent.set_train_mode(False)

        warmup_hands = min(5, self.hands_per_session - 1)
        for _ in range(warmup_hands):
            self.session.new_hand()
            while not self.session.is_finished:
                current_player = self.session.current_player
                if current_player == 0:
                    obs = self.session.get_observation(viewer_id=0)
                    action = self.agent.select_action(obs)
                else:
                    obs = self.session.get_observation(viewer_id=1)
                    action = opponent.select_action(obs)
                if isinstance(action, tuple):
                    action = action[0]
                self.session.step(action)

        self.session.new_hand()
        while not self.session.is_finished:
            current_player = self.session.current_player
            if current_player == 0:
                obs = self.session.get_observation(viewer_id=0)
                evaluations = self.agent.get_action_evaluations(obs)
                selected_eval = max(evaluations, key=lambda item: item["value"])
                action = selected_eval["action"]
                episode_trace.append(
                    {
                        "type": "agent_action",
                        "opponent": self._get_current_opponent_name(),
                        "observation": obs.to_dict(),
                        "evaluations": [
                            {
                                "action": item["action"].name,
                                "value": item["value"],
                                "base_value": item["base_value"],
                                "delta": item["delta"],
                                "gate": item["gate"],
                            }
                            for item in evaluations
                        ],
                        "selected_action": action.name,
                    }
                )
            else:
                opponent_obs = self.session.get_observation(viewer_id=1)
                agent_view_obs = self.session.get_observation(viewer_id=0)
                state_encoding = self.agent.encode_observation(agent_view_obs, viewer_id=0)
                stats_vec = self.agent.encode_macro_stats(agent_view_obs)
                probs = self.agent.predict_action_probs(state_encoding, stats_vec).squeeze(0)
                action = opponent.select_action(opponent_obs)
                episode_trace.append(
                    {
                        "type": "opponent_action",
                        "opponent": self._get_current_opponent_name(),
                        "observation": agent_view_obs.to_dict(),
                        "predicted_probs": {
                            "FOLD": round(probs[0].item(), 4),
                            "CALL": round(probs[1].item(), 4),
                            "RAISE": round(probs[2].item(), 4),
                        },
                        "actual_action": action.name,
                    }
                )
            if isinstance(action, tuple):
                action = action[0]
            self.session.step(action)

        rewards = self.session.game.get_reward()
        self.agent.set_train_mode(old_train_mode)
        return {
            "trace": episode_trace,
            "final_rewards": rewards,
            "opponent": self._get_current_opponent_name(),
        }

    def update_params(self, params: Dict):
        if "lr" in params:
            lr = params["lr"]
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = lr
            print(f"Learning rate updated to: {lr}")
        if "hands_per_session" in params:
            self.hands_per_session = params["hands_per_session"]
            print(f"Hands per session updated to: {self.hands_per_session}")
        if "action_loss_weight" in params:
            self.action_loss_weight = params["action_loss_weight"]
            print(f"Action loss weight updated to: {self.action_loss_weight}")
        if "gate_target" in params:
            self.gate_target = params["gate_target"]
            print(f"Gate target updated to: {self.gate_target}")
        if "gate_reg_weight" in params:
            self.gate_reg_weight = params["gate_reg_weight"]
            print(f"Gate reg weight updated to: {self.gate_reg_weight}")
        if "modulation_reg_weight" in params:
            self.modulation_reg_weight = params["modulation_reg_weight"]
            print(f"Modulation reg weight updated to: {self.modulation_reg_weight}")
        if "rotate_every" in params:
            self.rotate_every = params["rotate_every"]
            print(f"Rotate every updated to: {self.rotate_every}")
        if "snapshot_every" in params:
            self.snapshot_every = params["snapshot_every"]
            print(f"Snapshot every updated to: {self.snapshot_every}")
