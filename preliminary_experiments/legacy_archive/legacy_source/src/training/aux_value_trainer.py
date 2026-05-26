import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Tuple
from src.engine.leduc_game import LeducGame, Action
from src.agents.base import BaseAgent
from src.training.base import BaseTrainer


class AuxValueTrainer(BaseTrainer):
    """
    Self-play trainer for AuxValueAgent.

    Adds a pre-action Bellman consistency auxiliary loss on top of standard
    TD(0) training:

        Main loss:  V(post_t) → V(post_{t+1})  or  terminal reward
        Aux loss:   V(pre_t)  → max_a V(post_a)  (detached)

    The auxiliary loss provides extra gradient signal at each decision step
    without changing the TD bootstrap chain. aux_weight controls its scale.
    """

    def __init__(self, agent: BaseAgent, learning_rate: float = 1e-4, aux_weight: float = 0.5):
        super().__init__(agent, eval_interval=50, eval_num_games=100)
        self.optimizer = optim.Adam(self.agent.model.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()
        self.game = LeducGame()
        self.aux_weight = aux_weight

    def collect_episode(self) -> Tuple:
        """
        Play one episode.

        Returns (chains, pre_action_data, rewards):
          chains[p]           — list of chosen post-action encoded tensors (TD chain)
          pre_action_data[p]  — list of (pre_encoded, [post_encoded, ...]) per step
          rewards             — [r0, r1]
        """
        self.game.reset()
        chains = [[], []]
        pre_action_data = [[], []]

        while not self.game.is_finished:
            current_player = self.game.current_player
            obs = self.game.get_observation(viewer_id=current_player)

            # Pre-action encoding (before any action is committed)
            pre_encoded = self.agent.encode_observation(obs, viewer_id=current_player)

            # Evaluate all legal actions once — reused for both action selection and aux task
            evaluations = self.agent.get_action_evaluations(obs)
            post_encodeds = [e["encoded"] for e in evaluations]
            pre_action_data[current_player].append((pre_encoded, post_encodeds))

            # Select action via Boltzmann (train) or greedy (eval), replicating
            # select_action logic to avoid a redundant second call to get_action_evaluations
            values = torch.tensor([e["value"] for e in evaluations])
            if self.agent.train_mode:
                probs = torch.softmax(values / self.agent.temperature, dim=0)
                idx = torch.multinomial(probs, 1).item()
            else:
                idx = int(values.argmax().item())

            selected_eval = evaluations[idx]
            action = selected_eval["action"]

            # Record chosen post-action encoded state for the TD chain
            chains[current_player].append(selected_eval["encoded"])

            self.game.step(action)

        rewards = self.game.get_reward()
        return chains, pre_action_data, rewards

    def update_model(self, batch_data: list) -> float:
        """
        Compute and apply gradients for both losses.

        Main TD(0) loss: same as SelfPlayTrainer.
        Auxiliary loss:  V(pre) → max_a V(post_a), weighted by self.aux_weight.
        """
        self.optimizer.zero_grad()
        total_losses = []

        for chains, pre_action_data, rewards in batch_data:
            for p_idx in [0, 1]:
                chain = chains[p_idx]
                if not chain:
                    continue

                # --- Main TD(0) loss ---
                for t in range(len(chain)):
                    prediction = self.agent.model(chain[t]).squeeze(0)

                    if t == len(chain) - 1:
                        target = torch.FloatTensor([rewards[p_idx]])
                    else:
                        with torch.no_grad():
                            target = self.agent.model(chain[t + 1]).squeeze(0)

                    total_losses.append(self.criterion(prediction, target))

                # --- Auxiliary pre-action Bellman consistency loss ---
                for pre_encoded, post_encodeds in pre_action_data[p_idx]:
                    # Best post-action value under current network (detached: target, not co-trained)
                    with torch.no_grad():
                        post_vals = torch.stack([
                            self.agent.model(enc).squeeze() for enc in post_encodeds
                        ])
                        best_post_val = post_vals.max().unsqueeze(0)

                    pre_val = self.agent.model(pre_encoded).squeeze(0)
                    aux_loss = self.criterion(pre_val, best_post_val) * self.aux_weight
                    total_losses.append(aux_loss)

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
        if "lr" in params:
            new_lr = params["lr"]
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = new_lr
            print(f"Learning rate updated to: {new_lr}")
        if "aux_weight" in params:
            self.aux_weight = params["aux_weight"]
            print(f"aux_weight updated to: {self.aux_weight}")
