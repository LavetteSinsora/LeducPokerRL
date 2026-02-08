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
            "win_rate": 0.0
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
            self.current_stats["win_rate"] = data["win_rate"]
            self.history.append({
                "episode": data["episode"],
                "win_rate": data["win_rate"],
                "type": "win_rate"
            })
        
        # Trim history if too long
        if len(self.history) > self.max_history:
            self.history = self.history[-self.max_history:]

    def start_training(self, episodes: int = 1000, batch_size: int = 32, lr: float = 1e-4):
        if self.is_training:
            return False
        
        self.is_training = True
        self.trainer.optimizer.param_groups[0]['lr'] = lr
        
        def run_target():
            try:
                self.trainer.train(
                    num_episodes=episodes,
                    batch_size=batch_size,
                    save_path=self.model_path,
                    callback=self._training_callback
                )
            finally:
                self.is_training = False

        self.training_thread = threading.Thread(target=run_target, daemon=True)
        self.training_thread.start()
        return True

    def stop_training(self):
        if not self.is_training:
            return False
        
        self.trainer.request_stop()
        return True

    def get_status(self):
        return {
            "is_training": self.is_training,
            "stats": self.current_stats
        }

    def get_history(self):
        return self.history
