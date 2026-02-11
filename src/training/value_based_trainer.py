import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Tuple
from src.engine.leduc_game import LeducGame, Action
from src.agents.base import BaseAgent
from src.training.base import BaseTrainer

class SelfPlayTrainer(BaseTrainer):
    def __init__(self, agent: BaseAgent, learning_rate=1e-4):
        super().__init__(agent, eval_interval=50, eval_num_games=100)
        self.optimizer = optim.Adam(self.agent.model.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()
        self.game = LeducGame()

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
        """TD(0) on per-player post-action state chains.
        Each chain: V(s_t) -> V(s_{t+1}), last state -> terminal reward."""
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
                        # Last action -> target is terminal reward
                        target = torch.FloatTensor([rewards[p_idx]])
                    else:
                        # Bootstrap from next state in this player's chain
                        with torch.no_grad():
                            target = self.agent.model(chain[t + 1]).squeeze(0)

                    loss = self.criterion(prediction, target)
                    total_losses.append(loss)

        if total_losses:
            mean_loss = torch.stack(total_losses).mean()
            mean_loss.backward()
            self.optimizer.step()
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
        self.agent.set_train_mode(False) # Use greedy for debug

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
            "final_rewards": rewards
        }

    def update_params(self, params: Dict):
        """Updates the learning rate of the optimizer."""
        if "lr" in params:
            new_lr = params["lr"]
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = new_lr
            print(f"Learning rate updated to: {new_lr}")

if __name__ == "__main__":
    import argparse
    import os
    from src.agents.value_based import ValueBasedAgent

    parser = argparse.ArgumentParser(description="Train a ValueBasedAgent through self-play.")
    parser.add_argument("--episodes", type=int, default=1000, help="Number of episodes to train.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for updates.")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate.")
    parser.add_argument("--save_path", type=str, default="models/value_agent.pt", help="Path to save the trained model.")

    args = parser.parse_args()

    # Initialize agent and trainer
    agent = ValueBasedAgent()
    trainer = SelfPlayTrainer(agent, learning_rate=args.lr)

    print(f"Starting training for {args.episodes} episodes...")
    trainer.train(num_episodes=args.episodes, batch_size=args.batch_size, save_path=args.save_path)
    print("Training complete.")
