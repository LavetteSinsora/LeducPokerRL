import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Tuple
from src.engine.leduc_game import LeducGame, Action
from src.agents.base import BaseAgent
from src.training.base import BaseTrainer


class NStepValueTrainer(BaseTrainer):
    """
    Self-play trainer using n-step returns instead of TD(0).

    For a per-player post-action chain of length L:
      - If t + n_steps >= L: target_t = terminal reward  (no bootstrapping)
      - If t + n_steps <  L: target_t = V(s_{t+n_steps})  (bootstrap from n steps ahead)

    In poker there is no discounting (gamma=1) and rewards are only terminal,
    so the n-step return simplifies to either the terminal reward or a single
    bootstrap value.  With n=3 and typical Leduc chains of 2-4 steps, most
    transitions will use the actual terminal reward, giving cleaner gradients.
    """

    def __init__(self, agent: BaseAgent, learning_rate: float = 1e-4, n_steps: int = 3):
        super().__init__(agent, eval_interval=50, eval_num_games=100)
        self.optimizer = optim.Adam(self.agent.model.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()
        self.game = LeducGame()
        self.n_steps = n_steps

    def collect_episode(self) -> Tuple[List[List[torch.Tensor]], List[float]]:
        """Play one episode, returning per-player post-action state chains and rewards."""
        self.game.reset()
        chains = [[], []]  # chains[0] = P0's post-action states, chains[1] = P1's

        while not self.game.is_finished:
            current_player = self.game.current_player
            obs = self.game.get_observation(viewer_id=current_player)
            action = self.agent.select_action(obs)
            if isinstance(action, tuple):
                action = action[0]

            # Record post-action state for the acting player (with board masking)
            post_obs, _ = LeducGame.simulate_action(obs, action)
            encoded = self.agent.encode_observation(post_obs, viewer_id=current_player)
            chains[current_player].append(encoded)

            self.game.step(action)

        rewards = self.game.get_reward()
        return chains, rewards

    def update_model(self, batch_data: list) -> float:
        """N-step returns on per-player post-action state chains.

        For each timestep t in a chain of length L:
          - If t + n_steps >= L: target = terminal reward
          - Otherwise:           target = V(chain[t + n_steps]).detach()
        """
        self.optimizer.zero_grad()
        total_losses = []

        for chains, rewards in batch_data:
            for p_idx in [0, 1]:
                chain = chains[p_idx]
                if not chain:
                    continue

                L = len(chain)
                for t in range(L):
                    prediction = self.agent.model(chain[t]).squeeze(0)

                    if t + self.n_steps >= L:
                        # Within n steps of terminal: use actual reward
                        target = torch.FloatTensor([rewards[p_idx]])
                    else:
                        # Bootstrap from n steps ahead
                        with torch.no_grad():
                            target = self.agent.model(chain[t + self.n_steps]).squeeze(0)

                    loss = self.criterion(prediction, target)
                    total_losses.append(loss)

        if total_losses:
            mean_loss = torch.stack(total_losses).mean()
            mean_loss.backward()
            self.optimizer.step()
            return mean_loss.item()
        return 0.0

    def debug_episode(self) -> Dict:
        """Plays one greedy episode and records per-step action evaluations for the analyzer."""
        self.game.reset()
        episode_trace = []

        old_train_mode = self.agent.train_mode
        self.agent.set_train_mode(False)

        while not self.game.is_finished:
            current_player = self.game.current_player
            obs = self.game.get_observation(viewer_id=current_player)

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

        rewards = self.game.get_reward()
        for step in episode_trace:
            player_reward = rewards[step["player_id"]]
            step["true_value"] = player_reward
            pred_val = next(e["value"] for e in step["evaluations"] if e["action"] == step["selected_action"])
            step["prediction_error"] = (pred_val - player_reward) ** 2

        self.agent.set_train_mode(old_train_mode)
        return {
            "trace": episode_trace,
            "final_rewards": rewards,
            "eval_type": "value",
        }

    def update_params(self, params: Dict):
        """Updates trainer parameters: learning rate and n_steps."""
        if "lr" in params:
            new_lr = params["lr"]
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = new_lr
            print(f"Learning rate updated to: {new_lr}")
        if "n_steps" in params:
            self.n_steps = int(params["n_steps"])
            print(f"n_steps updated to: {self.n_steps}")
