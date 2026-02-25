import torch
from typing import Dict
from src.agents.base import BaseAgent
from src.engine.leduc_game import Action
from src.training.actor_critic_trainer import ActorCriticTrainer


class EntropyACTrainer(ActorCriticTrainer):
    """
    Actor-Critic trainer with entropy regularization.

    Extends ActorCriticTrainer by adding an entropy bonus to the loss:
        total_loss = policy_loss + value_coeff * value_loss - entropy_coeff * entropy

    The negative sign on entropy_coeff means we MAXIMIZE entropy, encouraging
    spread-out (mixed) strategies that are harder to exploit. This is especially
    useful in poker where deterministic strategies are easily countered.

    Entropy is computed as:
        H(pi) = -sum(pi(a|s) * log(pi(a|s))) for all legal actions

    To compute this properly, collect_episode() also records the full masked
    probability distribution at each step, not just the log_prob of the
    chosen action.
    """

    def __init__(self, agent: BaseAgent, learning_rate: float = 1e-3,
                 value_coeff: float = 0.5, entropy_coeff: float = 0.01):
        super().__init__(agent, learning_rate=learning_rate, value_coeff=value_coeff)
        self.entropy_coeff = entropy_coeff

    def collect_episode(self) -> dict:
        """Play one game of poker, recording log_probs, values, rewards, AND
        full masked probability distributions for entropy computation.

        Returns a dict with:
          - log_probs: per-player list of log-probabilities of chosen actions
          - values: per-player list of V(s) predictions at each decision point
          - rewards: final chips won/lost by each player
          - action_probs: per-player list of full masked probability tensors
        """
        self.game.reset()

        log_probs = [[], []]
        values = [[], []]
        action_probs = [[], []]  # full masked probability distributions

        self.agent.set_train_mode(True)

        while not self.game.is_finished:
            player = self.game.current_player
            obs = self.game.get_observation(viewer_id=player)
            encoded = self.agent.encode_observation(obs)

            # Get action probabilities AND value estimate from the network
            probs, value = self.agent.model(encoded)
            probs = probs.squeeze(0)
            value = value.squeeze(0)

            # Mask illegal actions
            legal_mask = torch.zeros(3)
            for action in obs.legal_actions:
                legal_mask[action.value] = 1.0
            probs = probs * legal_mask
            probs = probs / probs.sum()

            # Sample an action and record log-probability, value, AND full probs
            dist = torch.distributions.Categorical(probs)
            action_idx = dist.sample()
            log_probs[player].append(dist.log_prob(action_idx))
            values[player].append(value)
            action_probs[player].append(probs)

            self.game.step(Action(action_idx.item()))

        rewards = self.game.get_reward()
        return {
            "log_probs": log_probs,
            "values": values,
            "rewards": rewards,
            "action_probs": action_probs,
        }

    def update_model(self, batch_data: list) -> float:
        """Learn from a batch of games using REINFORCE with baseline + entropy bonus.

        For each game, for each player seat:
          advantage = reward - V(s).detach()
          policy_loss = -log_prob * advantage
          value_loss  = (V(s) - reward)^2
          entropy     = -sum(p * log(p + 1e-10))
          total_loss  = policy_loss + value_coeff * value_loss - entropy_coeff * entropy
        """
        self.optimizer.zero_grad()
        total_loss = torch.tensor(0.0)

        for episode in batch_data:
            log_probs = episode["log_probs"]
            values = episode["values"]
            rewards = episode["rewards"]
            action_probs = episode["action_probs"]

            for player in [0, 1]:
                if not log_probs[player]:
                    continue

                reward = rewards[player]

                for lp, v, ap in zip(log_probs[player], values[player],
                                     action_probs[player]):
                    # Advantage: was this outcome better or worse than expected?
                    advantage = reward - v.detach()

                    # Policy loss: REINFORCE with baseline
                    policy_loss = -lp * advantage

                    # Value loss: train critic to predict the actual outcome
                    value_loss = (v - reward) ** 2

                    # Entropy bonus: encourage exploration
                    # H(pi) = -sum(p * log(p + eps)) -- higher entropy = more spread out
                    entropy = -(ap * torch.log(ap + 1e-10)).sum()

                    # Subtract entropy_coeff * entropy to MAXIMIZE entropy
                    total_loss = (total_loss + policy_loss
                                  + self.value_coeff * value_loss
                                  - self.entropy_coeff * entropy)

        total_loss = total_loss / len(batch_data)
        total_loss.backward()
        self.optimizer.step()

        return total_loss.item()

    def debug_episode(self) -> Dict:
        """Plays one episode and records action probabilities, value estimates,
        and per-step entropy."""
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

            # Compute entropy of the masked distribution for this step
            probs_list = [e["probability"] for e in evaluations]
            probs_tensor = torch.tensor(probs_list)
            step_entropy = -(probs_tensor * torch.log(probs_tensor + 1e-10)).sum().item()

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
                "entropy": step_entropy,
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
            "eval_type": "entropy_ac",
        }

    def update_params(self, params: Dict):
        """Allow the dashboard to adjust lr, value_coeff, and entropy_coeff."""
        super().update_params(params)
        if "entropy_coeff" in params:
            self.entropy_coeff = params["entropy_coeff"]
            print(f"Entropy coefficient updated to: {self.entropy_coeff}")
