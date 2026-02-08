from abc import ABC, abstractmethod
from typing import Dict, Optional, Callable

class BaseTrainer(ABC):
    """
    Abstract base class for all trainers.
    Ensures a consistent interface for the TrainingManager.
    """
    
    @abstractmethod
    def train(self, num_episodes: int, batch_size: int = 32, save_path: str = None, callback: Optional[Callable] = None):
        """Standard training loop."""
        pass

    @abstractmethod
    def request_stop(self):
        """Signals the trainer to stop early."""
        pass

    @abstractmethod
    def update_params(self, params: Dict):
        """Updates trainer parameters (e.g., learning rate) while running."""
        pass
