import torch
import torch.nn as nn
import torch.optim as optim
import random
from dataclasses import dataclass
from typing import List, Tuple, Callable, Optional
import numpy as np
from src.engine.leduc_game import LeducGame, Action
from src.agents.value_based import ValueBasedAgent

@dataclass
class TrajectoryStep:
    encoded_state: torch.Tensor
    player_id: int

class SelfPlayTrainer:
    def __init__(self, agent: ValueBasedAgent, learning_rate=1e-4):
        self.agent = agent
        self.optimizer = optim.Adam(self.agent.model.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()
        self.game = LeducGame()
        self.stop_requested = False

    def train(self, num_episodes: int, batch_size: int = 32, save_path: str = None, callback: Optional[Callable] = None):
        self.agent.set_train_mode(True)
        self.stop_requested = False
        
        batch_data = []
        for episode in range(num_episodes):
            if self.stop_requested:
                print("Training stop requested.")
                break

            trajectory = self._play_episode()
            batch_data.append(trajectory)
            
            # Update the network once we've reached the batch size
            if len(batch_data) >= batch_size:
                loss = self._update_network(batch_data)
                batch_data = [] # Clear the batch
                
                if callback:
                    callback({
                        "episode": episode + 1,
                        "loss": loss,
                        "type": "batch_update"
                    })

                if (episode + 1) % 100 == 0 or episode < batch_size:
                    print(f"Episode {episode + 1}, Batch Loss: {loss:.4f}")

            # Periodically evaluate
            if (episode + 1) % 50 == 0:
                win_rate = self.evaluate(num_games=100)
                if callback:
                    callback({
                        "episode": episode + 1,
                        "win_rate": win_rate,
                        "type": "evaluation"
                    })
                print(f"Episode {episode + 1}, Evaluation Win Rate: {win_rate:.2f}")

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
        Returns the win rate (percentage of games where reward > 0).
        """
        from src.agents.heuristic import HeuristicAgent
        opponent = HeuristicAgent()
        
        self.agent.set_train_mode(False)
        wins = 0
        
        for _ in range(num_games):
            self.game.reset()
            # Randomize agent position (Player 0 or 1)
            agent_id = random.randint(0, 1)
            
            while not self.game.is_finished:
                curr_player = self.game.current_player
                obs = self.game.get_observation(viewer_id=curr_player)
                
                if curr_player == agent_id:
                    action = self.agent.select_action(obs)
                else:
                    action = opponent.select_action(obs)
                
                self.game.step(action)
            
            rewards = self.game.get_reward()
            if rewards[agent_id] > 0:
                wins += 1
            elif rewards[agent_id] == 0:
                wins += 0.5 # Count draw as half win
        
        self.agent.set_train_mode(True)
        return wins / num_games

    def request_stop(self):
        self.stop_requested = True

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
