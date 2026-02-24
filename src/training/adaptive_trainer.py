"""
Adaptive Trainer — session-based training for opponent-exploiting agents.

Extends SelfPlayTrainer to use PokerSession instead of LeducGame.
Each call to collect_episode() plays a full session of multiple hands,
accumulating opponent statistics across the session.

The key difference from SelfPlayTrainer:
  - SelfPlayTrainer plays one isolated hand per collect_episode()
  - AdaptiveTrainer plays a session of N hands, with stats accumulating

update_model() is inherited unchanged — it receives (chains, rewards) tuples
per hand, and the session boundary is invisible to TD(0). This is correct
because value targets are per-hand terminal rewards.
"""

import os
import torch
from dataclasses import replace
from typing import Dict, List, Tuple

from src.engine.leduc_game import LeducGame, Action
from src.engine.poker_session import PokerSession
from src.agents.base import BaseAgent
from src.training.value_based_trainer import SelfPlayTrainer


class AdaptiveTrainer(SelfPlayTrainer):

    def __init__(self, agent: BaseAgent, learning_rate: float = 1e-4,
                 hands_per_session: int = 30):
        super().__init__(agent, learning_rate=learning_rate)
        self.hands_per_session = hands_per_session
        self.session = PokerSession()

    def collect_episode(self) -> List[Tuple[List[List[torch.Tensor]], List[float]]]:
        """Play one session of hands, returning per-hand training data.

        Returns a list of (chains, rewards) tuples — one per hand in the session.
        Stats accumulate across hands within the session and reset between sessions.
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

                # Record post-action state for TD learning (with board masking).
                # Carry opponent_stats into simulated state so stat features
                # are meaningful in the stored chain tensors.
                post_obs, _ = LeducGame.simulate_action(obs, action)
                if obs.opponent_stats is not None:
                    post_obs = replace(post_obs, opponent_stats=obs.opponent_stats)
                encoded = self.agent.encode_observation(post_obs, viewer_id=current_player)
                chains[current_player].append(encoded)

                self.session.step(action)

            rewards = self.session.game.get_reward()
            session_data.append((chains, rewards))

        return session_data

    def train(self, num_episodes: int, batch_size: int = 32, save_path: str = None,
              callback=None, start_episode: int = 0):
        """Session-based training loop.

        Args:
            num_episodes: Number of sessions to run. Each session plays
                hands_per_session hands, so total hands = num_episodes * hands_per_session.
            batch_size: Number of individual hands to accumulate before updating.
            save_path: Where to persist the model on completion.
            callback: Progress callback (see BaseTrainer docstring).
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

            # Evaluate periodically (scale interval by hands_per_session)
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
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            self.agent.save_model(save_path)
            print(f"Model saved to {save_path}")

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
