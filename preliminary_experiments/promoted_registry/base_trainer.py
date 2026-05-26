"""Shared trainer base class for promoted agents."""

from abc import ABC, abstractmethod
from typing import Callable, Dict, Optional

class BaseTrainer(ABC):
    """
    Base class for all trainers.

    Provides a concrete training loop that calls two abstract hooks:
      - collect_episode()  — play one episode, return trajectory data
      - update_model()     — consume a batch, return scalar loss

    Subclasses that need a completely different loop can override train().
    """

    def __init__(self, agent, eval_interval: int = 50,
                 eval_num_games: int = 100, eval_opponent_factory=None):
        self.agent = agent
        self.eval_interval = eval_interval
        self.eval_num_games = eval_num_games
        self.eval_opponent_factory = eval_opponent_factory
        self.stop_requested = False

    # ------------------------------------------------------------------
    # Abstract hooks — subclasses MUST implement
    # ------------------------------------------------------------------

    @abstractmethod
    def collect_episode(self):
        """Play one episode and return trajectory data.

        The return type is up to the subclass (list of transitions,
        tuple of chains+rewards, etc.).
        """
        pass

    @abstractmethod
    def update_model(self, batch_data: list) -> float:
        """Consume a batch of trajectory data and return scalar loss."""
        pass

    @abstractmethod
    def update_params(self, params: Dict):
        """Updates trainer parameters (e.g., learning rate) while running."""
        pass

    # ------------------------------------------------------------------
    # Concrete defaults
    # ------------------------------------------------------------------

    def train(self, num_episodes: int, batch_size: int = 32, save_path: str = None,
              callback: Optional[Callable] = None, start_episode: int = 0):
        """Standard training loop.

        Args:
            num_episodes: How many episodes to run.
            batch_size: Episodes per network update.
            save_path: Where to persist the model on completion.
            callback: Progress callback (see module docstring).
            start_episode: Episode counter offset for resumed training.
        """
        self.agent.set_train_mode(True)
        self.stop_requested = False

        batch_data = []
        for i in range(num_episodes):
            if self.stop_requested:
                print("Training stop requested.")
                break

            episode = start_episode + i + 1  # Current episode number (1-indexed)
            trajectory = self.collect_episode()
            batch_data.append(trajectory)

            # Update the network once we've reached the batch size
            if len(batch_data) >= batch_size:
                loss = self.update_model(batch_data)
                batch_data = []  # Clear the batch

                if callback:
                    callback({
                        "episode": episode,
                        "loss": loss,
                        "type": "batch_update"
                    })

                if i < batch_size or (episode) % 100 == 0:
                    print(f"Episode {episode}, Batch Loss: {loss:.4f}")

            # Periodically evaluate
            if episode % self.eval_interval == 0:
                avg_chips = self.evaluate(num_games=self.eval_num_games)
                if callback:
                    callback({
                        "episode": episode,
                        "avg_chips_per_round": avg_chips,
                        "type": "evaluation"
                    })
                print(f"Episode {episode}, Avg Chips/Round: {avg_chips:+.2f}")

        if save_path:
            import os
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            self.agent.save_model(save_path)
            print(f"Model saved to {save_path}")

    def request_stop(self):
        """Signals the trainer to stop early."""
        self.stop_requested = True

    def evaluate(self, num_games: int = None) -> float:
        """Evaluate the agent against an opponent.

        Uses eval_opponent_factory if provided, otherwise defaults to
        HeuristicAgent.
        """
        if num_games is None:
            num_games = self.eval_num_games

        if self.eval_opponent_factory:
            opponent = self.eval_opponent_factory()
        else:
            from agents.heuristic.agent import HeuristicAgent

            opponent = HeuristicAgent()

        from agents.evaluation import quick_evaluate

        self.agent.set_train_mode(False)
        avg_chips = quick_evaluate(self.agent, opponent, num_rounds=num_games)
        self.agent.set_train_mode(True)

        return avg_chips

    def debug_episode(self) -> Dict:
        """Run a single episode in debug mode and return a trace dict.

        Subclasses that support the analyzer should override this.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement debug_episode(). "
            "Override this method to enable the analyzer's episode trace view."
        )
