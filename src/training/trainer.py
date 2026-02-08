import torch
import torch.nn as nn
import torch.optim as optim
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple, Callable, Optional
import numpy as np
from src.engine.leduc_game import LeducGame, Action
from src.agents.value_based import ValueBasedAgent
from src.agents.heuristic import HeuristicAgent
from src.training.base import BaseTrainer
from src.training.evaluation import quick_evaluate

@dataclass
class TrajectoryStep:
    encoded_state: torch.Tensor
    player_id: int

class SelfPlayTrainer(BaseTrainer):
    def __init__(self, agent: ValueBasedAgent, learning_rate=1e-4):
        self.agent = agent
        self.optimizer = optim.Adam(self.agent.model.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()
        self.game = LeducGame()
        self.stop_requested = False

    def train(self, num_episodes: int, batch_size: int = 32, save_path: str = None, callback: Optional[Callable] = None, start_episode: int = 0):
        self.agent.set_train_mode(True)
        self.stop_requested = False
        
        batch_data = []
        for i in range(num_episodes):
            if self.stop_requested:
                print("Training stop requested.")
                break

            episode = start_episode + i + 1  # Current episode number (1-indexed)
            trajectory = self._play_episode()
            batch_data.append(trajectory)
            
            # Update the network once we've reached the batch size
            if len(batch_data) >= batch_size:
                loss = self._update_network(batch_data)
                batch_data = [] # Clear the batch
                
                if callback:
                    callback({
                        "episode": episode,
                        "loss": loss,
                        "type": "batch_update"
                    })

                if i < batch_size or (episode) % 100 == 0:
                    print(f"Episode {episode}, Batch Loss: {loss:.4f}")

            # Periodically evaluate
            if episode % 50 == 0:
                avg_chips = self.evaluate(num_games=100)
                if callback:
                    callback({
                        "episode": episode,
                        "avg_chips_per_round": avg_chips,
                        "type": "evaluation"
                    })
                print(f"Episode {episode}, Avg Chips/Round: {avg_chips:+.2f}")

        if save_path and not self.stop_requested:
            import os
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(self.agent.model.state_dict(), save_path)
            print(f"Model saved to {save_path}")

    def _play_episode(self) -> Tuple[List[TrajectoryStep], List[float]]:
        self.game.reset()
        trajectories = []
        
        while not self.game.is_finished:
            current_player = self.game.current_player
            obs = self.game.get_observation(viewer_id=current_player)
            
            # Agent outputs action and the encoded state that was evaluated
            # In train_mode, select_action returns (action, encoded_state)
            action, encoded_state = self.agent.select_action(obs)
            
            if encoded_state is not None:
                trajectories.append(TrajectoryStep(
                    encoded_state=encoded_state,
                    player_id=current_player
                ))
            
            self.game.step(action)
        
        # Calculate final profits for both players from the game engine
        rewards = self.game.get_reward()
        return trajectories, rewards

    def _update_network(self, batch_data: List[Tuple[List[TrajectoryStep], List[float]]]) -> float:
        """
        Updates the network using all trajectories collected in a batch.
        """
        self.optimizer.zero_grad()
        
        total_losses = []
        for trajectories, rewards in batch_data:
            if not trajectories:
                continue
            
            for step in trajectories:
                target = torch.FloatTensor([rewards[step.player_id]])
                prediction = self.agent.model(step.encoded_state).squeeze(0)
                
                loss = self.criterion(prediction, target)
                total_losses.append(loss)
        
        if total_losses:
            # Average the loss across all decision points in the entire batch
            mean_loss = torch.stack(total_losses).mean()
            mean_loss.backward()
            self.optimizer.step()
            return mean_loss.item()
        
        return 0.0

    def evaluate(self, num_games: int = 100) -> float:
        """
        Evaluates the agent against a HeuristicAgent.
        Returns the average chips per round (more informative than win rate).
        
        Uses the decoupled evaluation module from src/training/evaluation.py.
        """
        opponent = HeuristicAgent()
        
        self.agent.set_train_mode(False)
        avg_chips = quick_evaluate(self.agent, opponent, num_rounds=num_games)
        self.agent.set_train_mode(True)
        
        return avg_chips

    def request_stop(self):
        self.stop_requested = True

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
