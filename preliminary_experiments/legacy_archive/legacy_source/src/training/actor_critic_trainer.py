import torch
import torch.optim as optim
from typing import Dict, List
from src.engine.leduc_game import LeducGame, Action
from src.agents.base import BaseAgent
from src.training.base import BaseTrainer


class ActorCriticTrainer(BaseTrainer):
    """
    Trains an ActorCriticAgent using REINFORCE with a learned value baseline.

    This is an incremental improvement over PolicyGradientTrainer. The key
    difference: instead of using raw reward as the reinforcement signal,
    we use advantage = reward - V(s), where V(s) is the critic's estimate
    of expected reward from state s.

    Each episode:
      1. Play a full game, recording log_probs, values, and rewards per player
      2. Compute advantage = reward - V(s).detach()  (was outcome better than expected?)
      3. Policy loss: -log_prob * advantage  (REINFORCE with baseline)
      4. Value loss:  MSE(V(s), reward)       (train the critic to predict outcomes)
      5. Total loss = policy_loss + value_coeff * value_loss
    """

    def __init__(self, agent: BaseAgent, learning_rate: float = 1e-3,
                 value_coeff: float = 0.5):
        super().__init__(agent, eval_interval=50, eval_num_games=100)
        self.optimizer = optim.Adam(self.agent.model.parameters(), lr=learning_rate)
        self.value_coeff = value_coeff
        self.game = LeducGame()

    def collect_episode(self) -> dict:
        """Play one game of poker and record everything that happened.

        Returns a dict with:
          - log_probs: per-player list of log-probabilities of chosen actions
          - values: per-player list of V(s) predictions at each decision point
          - rewards: final chips won/lost by each player
        """
        self.game.reset()

        # Collect per-player data since rewards differ by seat
        log_probs = [[], []]  # log_probs[player] = list of log_prob tensors
        values = [[], []]     # values[player] = list of V(s) tensors

        self.agent.set_train_mode(True)

        while not self.game.is_finished:
            player = self.game.current_player
            obs = self.game.get_observation(viewer_id=player)
            encoded = self.agent.encode_observation(obs)

            # Get action probabilities AND value estimate from the network
            probs, value = self.agent.model(encoded)
            probs = probs.squeeze(0)
            value = value.squeeze(0)   # shape: (1,)

            # Mask illegal actions
            legal_mask = torch.zeros(3)
            for action in obs.legal_actions:
                legal_mask[action.value] = 1.0
            probs = probs * legal_mask
            probs = probs / probs.sum()

            # Sample an action and record log-probability + value
            dist = torch.distributions.Categorical(probs)
            action_idx = dist.sample()
            log_probs[player].append(dist.log_prob(action_idx))
            values[player].append(value)

            self.game.step(Action(action_idx.item()))

        rewards = self.game.get_reward()  # [player_0_chips, player_1_chips]
        return {"log_probs": log_probs, "values": values, "rewards": rewards}

    def update_model(self, batch_data: list) -> float:
        """Learn from a batch of games using REINFORCE with a learned baseline.

        For each game, for each player seat:
          advantage = reward - V(s).detach()   (no gradient through baseline for policy)
          policy_loss = -log_prob * advantage
          value_loss  = (V(s) - reward)^2
          total_loss  = policy_loss + value_coeff * value_loss
        """
        self.optimizer.zero_grad()
        total_loss = torch.tensor(0.0)

        for episode in batch_data:
            log_probs = episode["log_probs"]
            values = episode["values"]
            rewards = episode["rewards"]

            for player in [0, 1]:
                if not log_probs[player]:
                    continue

                reward = rewards[player]

                for lp, v in zip(log_probs[player], values[player]):
                    # Advantage: was this outcome better or worse than expected?
                    # .detach() prevents policy gradients from flowing through V(s)
                    advantage = reward - v.detach()

                    # Policy loss: REINFORCE with baseline
                    policy_loss = -lp * advantage

                    # Value loss: train critic to predict the actual outcome
                    value_loss = (v - reward) ** 2

                    total_loss = total_loss + policy_loss + self.value_coeff * value_loss

        total_loss = total_loss / len(batch_data)
        total_loss.backward()
        self.optimizer.step()

        return total_loss.item()

    def debug_episode(self) -> Dict:
        """Plays one episode and records action probabilities + value estimates."""
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
                        "value_estimate": e["value_estimate"],
                    }
                    for e in evaluations
                ],
                "selected_action": action.name,
                "selected_action_id": action.value,
                "value_estimate": selected_eval["value_estimate"],
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
            "eval_type": "actor_critic",
        }

    def update_params(self, params: Dict):
        """Allow the dashboard to adjust learning rate and value_coeff during training."""
        if "lr" in params:
            new_lr = params["lr"]
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = new_lr
            print(f"Learning rate updated to: {new_lr}")
        if "value_coeff" in params:
            self.value_coeff = params["value_coeff"]
            print(f"Value coefficient updated to: {self.value_coeff}")
