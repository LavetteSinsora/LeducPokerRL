import threading
import os
import torch
from typing import List, Dict, Optional
from src.training.trainer import SelfPlayTrainer
from src.agents.value_based import ValueBasedAgent

class TrainingManager:
    def __init__(self, model_path: str = "models/value_agent.pt"):
        self.model_path = model_path
        self.agent = ValueBasedAgent()
        if os.path.exists(self.model_path):
            try:
                self.agent.model.load_state_dict(torch.load(self.model_path))
                print(f"Loaded existing model from {self.model_path}")
            except Exception as e:
                print(f"Error loading model: {e}")
        
        self.trainer = SelfPlayTrainer(self.agent)
        self.training_thread: Optional[threading.Thread] = None
        self.is_training = False
        
        # Metrics storage
        self.history: List[Dict] = []
        self.max_history = 1000
        self.current_stats = {
            "episode": 0,
            "loss": 0.0,
            "avg_chips_per_round": 0.0
        }

    def _training_callback(self, data: Dict):
        if data["type"] == "batch_update":
            self.current_stats["episode"] = data["episode"]
            self.current_stats["loss"] = data["loss"]
            self.history.append({
                "episode": data["episode"],
                "loss": data["loss"],
                "type": "loss"
            })
        elif data["type"] == "evaluation":
            self.current_stats["episode"] = data["episode"]
            self.current_stats["avg_chips_per_round"] = data["avg_chips_per_round"]
            self.history.append({
                "episode": data["episode"],
                "avg_chips_per_round": data["avg_chips_per_round"],
                "type": "avg_chips"
            })
        
        # Trim history if too long
        if len(self.history) > self.max_history:
            self.history = self.history[-self.max_history:]

    def start_training(self, episodes: int = 1000, batch_size: int = 32, lr: float = 1e-4):
        """Start fresh training or resume if already training (to update LR)."""
        if self.is_training:
            # If already training, we can update the LR on the fly!
            self.trainer.update_params({"lr": lr})
            return True
        
        self.is_training = True
        self.trainer.update_params({"lr": lr})
        
        # Get starting episode from current stats (for resume functionality)
        start_episode = self.current_stats.get("episode", 0)
        
        def run_target():
            try:
                self.trainer.train(
                    num_episodes=episodes,
                    batch_size=batch_size,
                    save_path=self.model_path,
                    callback=self._training_callback,
                    start_episode=start_episode
                )
            finally:
                self.is_training = False

        self.training_thread = threading.Thread(target=run_target, daemon=True)
        self.training_thread.start()
        return True

    @property
    def has_training_history(self) -> bool:
        """Returns True if there is any training history (for resume detection)."""
        return len(self.history) > 0

    def stop_training(self):
        if not self.is_training:
            return False
        
        self.trainer.request_stop()
        return True

    def reset_agent(self):
        """Stops training, deletes model, and resets agent weights."""
        if self.is_training:
            self.stop_training()
            if self.training_thread:
                self.training_thread.join(timeout=2.0)
        
        # Reset agent and trainer
        self.agent = ValueBasedAgent()
        self.trainer = SelfPlayTrainer(self.agent)
        
        # Clear metrics
        self.history = []
        self.current_stats = {
            "episode": 0,
            "loss": 0.0,
            "avg_chips_per_round": 0.0
        }
        
        # Delete model file if it exists
        if os.path.exists(self.model_path):
            os.remove(self.model_path)
            print(f"Deleted model file: {self.model_path}")
        
        return True

    def get_status(self):
        return {
            "is_training": self.is_training,
            "has_history": self.has_training_history,
            "stats": self.current_stats
        }

    def get_history(self):
        return self.history

    def run_debug_episode(self) -> Dict:
        """Runs a single episode in debug mode and returns the trace."""
        return self.trainer.debug_episode()
