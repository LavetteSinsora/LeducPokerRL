import torch
import torch.optim as optim
from typing import Dict, List
from src.engine.leduc_game import LeducGame, Action
from src.agents.base import BaseAgent
from src.training.base import BaseTrainer


class PolicyGradientTrainer(BaseTrainer):
    """
    Trains a PolicyGradientAgent using the REINFORCE algorithm.

    Each episode:
      1. Play a full game, recording every (state, action) pair
      2. Use the final reward to compute the policy gradient loss
      3. Update the network to make winning actions more likely
    """

    def __init__(self, agent: BaseAgent, learning_rate: float = 1e-3):
        super().__init__(agent, eval_interval=50, eval_num_games=100)
        self.optimizer = optim.Adam(self.agent.model.parameters(), lr=learning_rate)
        self.game = LeducGame()

    def collect_episode(self) -> dict:
        """Play one game of poker and record everything that happened.

        Returns a dict with:
          - log_probs: the log-probability of each action the agent chose
          - reward: the final chips won/lost by the agent (playing as both players)
        """
        self.game.reset()

        # We'll collect log-probs separately for each player seat,
        # since the reward is different for each side
        log_probs = [[], []]  # log_probs[0] = player 0's actions, etc.

        self.agent.set_train_mode(True)

        while not self.game.is_finished:
            player = self.game.current_player
            obs = self.game.get_observation(viewer_id=player)
            encoded = self.agent.encode_observation(obs)

            # Get action probabilities from the policy network
            probs = self.agent.model(encoded).squeeze(0)

            # Mask illegal actions
            legal_mask = torch.zeros(3)
            for action in obs.legal_actions:
                legal_mask[action.value] = 1.0
            probs = probs * legal_mask
            probs = probs / probs.sum()

            # Sample an action and record its log-probability
            dist = torch.distributions.Categorical(probs)
            action_idx = dist.sample()
            log_probs[player].append(dist.log_prob(action_idx))

            self.game.step(Action(action_idx.item()))

        rewards = self.game.get_reward()  # [player_0_chips, player_1_chips]
        return {"log_probs": log_probs, "rewards": rewards}

    def update_model(self, batch_data: list) -> float:
        """Learn from a batch of games using the REINFORCE algorithm.

        For each game, for each player seat:
          loss = -sum(log_prob * reward)

        This pushes the policy toward actions that led to positive rewards
        and away from actions that led to negative rewards.
        """
        self.optimizer.zero_grad()
        total_loss = torch.tensor(0.0)

        for episode in batch_data:
            log_probs = episode["log_probs"]
            rewards = episode["rewards"]

            for player in [0, 1]:
                if not log_probs[player]:
                    continue

                reward = rewards[player]
                # REINFORCE: loss = -log_prob * reward
                # Negative because we want to maximize reward (gradient ascent)
                for lp in log_probs[player]:
                    total_loss = total_loss - lp * reward

        total_loss = total_loss / len(batch_data)
        total_loss.backward()
        self.optimizer.step()

        return total_loss.item()

    def debug_episode(self) -> Dict:
        """Plays one episode and records action probabilities at each step."""
        self.game.reset()
        episode_trace = []

        old_train_mode = self.agent.train_mode
        self.agent.set_train_mode(False)

        while not self.game.is_finished:
            current_player = self.game.current_player
            obs = self.game.get_observation(viewer_id=current_player)

            evaluations = self.agent.get_action_evaluations(obs)

            # Greedy: pick the highest-probability action
            selected_eval = max(evaluations, key=lambda x: x["probability"])
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
                        "action_id": e["action"].value,
                        "probability": e["probability"],
                        "raw_probability": e["raw_probability"],
                    }
                    for e in evaluations
                ],
                "selected_action": action.name,
                "selected_action_id": action.value,
                "encoded_state": selected_eval["encoded"].squeeze(0).tolist(),
            }

            episode_trace.append(step_info)
            self.game.step(action)

        rewards = self.game.get_reward()

        for step in episode_trace:
            step["true_value"] = rewards[step["player_id"]]

        self.agent.set_train_mode(old_train_mode)
        return {
            "trace": episode_trace,
            "final_rewards": rewards,
            "eval_type": "policy",
        }

    def update_params(self, params: Dict):
        """Allow the dashboard to adjust learning rate during training."""
        if "lr" in params:
            new_lr = params["lr"]
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = new_lr
            print(f"Learning rate updated to: {new_lr}")
