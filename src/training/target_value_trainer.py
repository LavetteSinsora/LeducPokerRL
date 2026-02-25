import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Tuple
from src.engine.leduc_game import LeducGame, Action
from src.training.value_based_trainer import SelfPlayTrainer


class TargetValueTrainer(SelfPlayTrainer):
    """
    TD(0) trainer using a frozen target network for stable bootstrap targets.

    Instead of: target = model(s_{t+1})        [moving target]
    Uses:       target = target_model(s_{t+1})  [stable target]

    The target network is synced to the main model every target_sync_every
    gradient steps.
    """

    def __init__(self, agent, learning_rate=1e-4, target_sync_every=100):
        super().__init__(agent, learning_rate=learning_rate)
        self.target_sync_every = target_sync_every
        self.gradient_steps = 0

    def update_model(self, batch_data):
        """TD(0) with target network for bootstrap values."""
        self.optimizer.zero_grad()
        total_losses = []

        for chains, rewards in batch_data:
            for p_idx in [0, 1]:
                chain = chains[p_idx]
                if not chain:
                    continue

                for t in range(len(chain)):
                    prediction = self.agent.model(chain[t]).squeeze(0)

                    if t == len(chain) - 1:
                        # Terminal: target is actual reward
                        target = torch.FloatTensor([rewards[p_idx]])
                    else:
                        # Bootstrap from TARGET network (not main model!)
                        with torch.no_grad():
                            target = self.agent.target_model(chain[t + 1]).squeeze(0)

                    loss = self.criterion(prediction, target)
                    total_losses.append(loss)

        if total_losses:
            mean_loss = torch.stack(total_losses).mean()
            mean_loss.backward()
            self.optimizer.step()

            self.gradient_steps += 1
            if self.gradient_steps % self.target_sync_every == 0:
                self.agent.sync_target()
                print(f"  Target network synced (step {self.gradient_steps})")

            return mean_loss.item()
        return 0.0

    def debug_episode(self) -> List[Dict]:
        """
        Plays one episode and records detailed mental simulations and final rewards.
        """
        self.game.reset()
        episode_trace = []

        # We temporarily disable Boltzmann exploration for cleaner debug traces
        old_train_mode = self.agent.train_mode
        self.agent.set_train_mode(False)  # Use greedy for debug

        while not self.game.is_finished:
            current_player = self.game.current_player
            obs = self.game.get_observation(viewer_id=current_player)

            # Get full simulation results
            evaluations = self.agent.get_action_evaluations(obs)

            # Select action greedily for the trace
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
                "evaluations": [
                    {
                        "action": e["action"].name,
                        "value": e["value"],
                        "action_id": e["action"].value,
                        "encoded_state": e["encoded"].squeeze(0).tolist()
                    } for e in evaluations
                ],
                "selected_action": action.name,
                "selected_action_id": action.value,
                "encoded_state": selected_eval["encoded"].squeeze(0).tolist()
            }

            episode_trace.append(step_info)
            self.game.step(action)

        # Calculate final profits
        rewards = self.game.get_reward()

        # Post-process trace to add "True Value" and "Prediction Error"
        for step in episode_trace:
            player_reward = rewards[step["player_id"]]
            step["true_value"] = player_reward

            # Prediction error for the selected action
            pred_val = next(e["value"] for e in step["evaluations"] if e["action"] == step["selected_action"])
            step["prediction_error"] = (pred_val - player_reward) ** 2

        self.agent.set_train_mode(old_train_mode)
        return {
            "trace": episode_trace,
            "final_rewards": rewards,
            "eval_type": "value",
        }

    def update_params(self, params: Dict):
        """Support lr and target_sync_every updates."""
        if "lr" in params:
            for pg in self.optimizer.param_groups:
                pg["lr"] = params["lr"]
        if "target_sync_every" in params:
            self.target_sync_every = params["target_sync_every"]
