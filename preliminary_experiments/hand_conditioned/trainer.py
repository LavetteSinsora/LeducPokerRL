"""Trainer for the hand-conditioned opponent action model."""

import json
import os
import random
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from agents.registry import registry
from engine.poker_session import PokerSession
from experiments.hand_conditioned_action_model_v1.agent import (
    HandConditionedActionModel,
    initialize_belief,
    update_belief_with_board,
)


DEFAULT_TRAIN_POOL = (
    "heuristic",
    "value_based",
    "adaptive_value",
    "modulated_value",
    "cfr",
    "entropy_ac",
    "belief_modulated",
)

DEFAULT_EVAL_POOL = ("heuristic", "value_based", "adaptive_value", "modulated_value", "cfr")


def load_registry_agent(agent_id: str):
    agent = registry.create(agent_id)
    checkpoint_path = registry.get_checkpoint_path(agent_id)
    if checkpoint_path and os.path.exists(checkpoint_path):
        try:
            agent.load_model(checkpoint_path)
        except Exception as exc:
            print(f"Warning: could not load checkpoint for {agent_id}: {exc}")
    agent.set_train_mode(False)
    return agent


class HandConditionedActionModelTrainer:
    """Population data generator and supervised trainer."""

    def __init__(
        self,
        model: HandConditionedActionModel,
        learning_rate: float = 3e-4,
        hands_per_session: int = 30,
        train_pool_ids: Sequence[str] = DEFAULT_TRAIN_POOL,
        eval_pool_ids: Sequence[str] = DEFAULT_EVAL_POOL,
        seed: int = 0,
    ):
        self.model = model
        self.hands_per_session = hands_per_session
        self.rng = random.Random(seed)
        self.session = PokerSession()

        self.optimizer = optim.Adam(self.model.model.parameters(), lr=learning_rate)
        self.criterion = nn.NLLLoss()

        self.train_agents = {agent_id: load_registry_agent(agent_id) for agent_id in train_pool_ids}
        self.eval_pool_ids = tuple(eval_pool_ids)

        self.sessions_played = 0
        self.last_metrics = {"loss": 0.0, "batch_accuracy": 0.0}

    def sample_pair(self) -> Tuple[str, str]:
        ids = list(self.train_agents)
        a = self.rng.choice(ids)
        b = self.rng.choice(ids)
        while b == a and len(ids) > 1:
            b = self.rng.choice(ids)
        return a, b

    def collect_session(self) -> dict:
        """Collect supervised action examples from one multi-hand session."""
        pair = self.sample_pair()
        agents = [self.train_agents[pair[0]], self.train_agents[pair[1]]]
        self.session.reset()

        examples = []
        for _ in range(self.hands_per_session):
            self.session.new_hand()

            while not self.session.is_finished:
                actor = self.session.current_player
                viewer = 1 - actor
                actor_obs = self.session.get_observation(viewer_id=actor)
                observer_obs = self.session.get_observation(viewer_id=viewer)
                action = agents[actor].select_action(actor_obs)
                if isinstance(action, tuple):
                    action = action[0]

                examples.append(
                    {
                        "obs": observer_obs,
                        "viewer_id": viewer,
                        "candidate_hand": self.session.game.player_hands[actor],
                        "action_idx": int(action),
                        "actor_id": actor,
                        "pair": pair,
                    }
                )
                self.session.step(action)

        self.sessions_played += 1
        return {"pair": pair, "examples": examples}

    def update_model(self, batch_data: list) -> float:
        self.optimizer.zero_grad()

        inputs = []
        targets = []
        for session_data in batch_data:
            for example in session_data["examples"]:
                inputs.append(
                    self.model.encode_example(
                        example["obs"],
                        viewer_id=example["viewer_id"],
                        candidate_hand=example["candidate_hand"],
                    ).squeeze(0)
                )
                targets.append(example["action_idx"])

        if not inputs:
            self.last_metrics = {"loss": 0.0, "batch_accuracy": 0.0}
            return 0.0

        batch_inputs = torch.stack(inputs)
        batch_targets = torch.tensor(targets, dtype=torch.long)
        log_probs = self.model.model(batch_inputs)
        loss = self.criterion(log_probs, batch_targets)
        loss.backward()
        self.optimizer.step()

        predictions = log_probs.argmax(dim=-1)
        batch_accuracy = (predictions == batch_targets).float().mean().item()
        self.last_metrics = {"loss": loss.item(), "batch_accuracy": batch_accuracy}
        return loss.item()

    def evaluate(self, sessions_per_opponent: int = 4, hands_per_session: int = 10) -> Dict:
        """Quick belief-quality eval used during training."""
        probe = load_registry_agent("modulated_value")
        metrics = []

        for target_id in self.eval_pool_ids:
            target_agent = load_registry_agent(target_id)
            pairings = [(target_agent, probe), (probe, target_agent)]
            belief_acc = []
            action_acc = []

            for left, right in pairings:
                for _ in range(sessions_per_opponent):
                    session = PokerSession()
                    beliefs = [None, None]
                    board_seen = [None, None]

                    for _ in range(hands_per_session):
                        session.new_hand()
                        beliefs = [
                            initialize_belief(session.game.player_hands[0]),
                            initialize_belief(session.game.player_hands[1]),
                        ]
                        board_seen = [None, None]

                        while not session.is_finished:
                            actor = session.current_player
                            viewer = 1 - actor
                            agents = [left, right]
                            actor_obs = session.get_observation(viewer_id=actor)
                            observer_obs = session.get_observation(viewer_id=viewer)
                            action = agents[actor].select_action(actor_obs)
                            if isinstance(action, tuple):
                                action = action[0]

                            if agents[actor] is target_agent:
                                if observer_obs.board is not None and board_seen[viewer] is None:
                                    beliefs[viewer] = update_belief_with_board(
                                        beliefs[viewer],
                                        viewer_hand=observer_obs.player_hand,
                                        board=observer_obs.board,
                                    )
                                    board_seen[viewer] = observer_obs.board

                                true_hand = session.game.player_hands[actor]
                                log_probs = self.model.predict_log_probs(observer_obs, viewer, true_hand).squeeze(0)
                                action_acc.append(float(int(log_probs.argmax().item() == int(action))))
                                beliefs[viewer] = self.model.update_belief(beliefs[viewer], observer_obs, viewer, action)
                                belief_acc.append(self.model.belief_top1_correct(beliefs[viewer], true_hand))

                            session.step(action)

            metrics.append(
                {
                    "target": target_id,
                    "belief_top1_accuracy": sum(belief_acc) / len(belief_acc) if belief_acc else 0.0,
                    "action_accuracy": sum(action_acc) / len(action_acc) if action_acc else 0.0,
                }
            )

        mean_belief_acc = sum(m["belief_top1_accuracy"] for m in metrics) / len(metrics)
        mean_action_acc = sum(m["action_accuracy"] for m in metrics) / len(metrics)
        return {
            "mean_belief_top1_accuracy": mean_belief_acc,
            "mean_action_accuracy": mean_action_acc,
            "per_opponent": metrics,
        }

    def train(
        self,
        num_sessions: int,
        batch_size: int = 32,
        save_path: str = None,
        callback=None,
        start_session: int = 0,
    ):
        self.model.set_train_mode(True)
        batch_data = []

        for session_idx in range(num_sessions):
            session_number = start_session + session_idx + 1
            batch_data.append(self.collect_session())

            if len(batch_data) >= batch_size:
                self.update_model(batch_data)
                batch_data = []

                if callback:
                    callback(
                        {
                            "session": session_number,
                            "hands_seen": session_number * self.hands_per_session,
                            "type": "batch_update",
                            **self.last_metrics,
                        }
                    )

                if session_number <= 3 or session_number % 50 == 0:
                    print(
                        f"Session {session_number}, Hands {session_number * self.hands_per_session}, "
                        f"Loss {self.last_metrics['loss']:.4f}, "
                        f"BatchAcc {self.last_metrics['batch_accuracy']:.4f}"
                    )

            if session_number % 200 == 0:
                metrics = self.evaluate()
                if callback:
                    callback(
                        {
                            "session": session_number,
                            "hands_seen": session_number * self.hands_per_session,
                            "type": "evaluation",
                            **metrics,
                        }
                    )
                print(
                    f"Session {session_number}, BeliefTop1 {metrics['mean_belief_top1_accuracy']:.3f}, "
                    f"ActionAcc {metrics['mean_action_accuracy']:.3f}"
                )

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            self.model.save_model(save_path)
            print(f"Model saved to {save_path}")
