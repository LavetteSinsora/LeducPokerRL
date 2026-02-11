import threading
import os
from typing import List, Dict, Optional
from src.agents.registry import registry

class TrainingManager:
    def __init__(self, agent_id: str = "value_based", model_path: Optional[str] = None):
        self.agent_id = agent_id
        
        # Use default path if none provided
        if model_path is None:
            self.model_path = f"models/{agent_id}_agent.pt"
        else:
            self.model_path = model_path
            
        # Initialize agent from registry
        self.agent = registry.create(agent_id)

        if os.path.exists(self.model_path):
            try:
                self.agent.load_model(self.model_path)
                print(f"Loaded existing model from {self.model_path}")
            except Exception as e:
                print(f"Error loading model: {e}")
        
        self.trainer = self._create_trainer()
        self.training_thread: Optional[threading.Thread] = None
        self.is_training = False
        
        # Metrics storage (no cap — kept in full for long-range trend analysis)
        self.history: List[Dict] = []
        self.current_stats = {
            "episode": 0,
            "loss": 0.0,
            "avg_chips_per_round": 0.0
        }

    def _create_trainer(self):
        """Factory method to create the appropriate trainer for the agent."""
        metadata = registry.get_metadata(self.agent_id)
        if metadata and metadata.trainer_class:
            return metadata.trainer_class(self.agent)
        raise ValueError(f"No trainer registered for agent type: {self.agent_id}")

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
        
        # No trim — full history preserved for long-range trend charts

    def start_training(self, episodes: int = 1000, batch_size: int = 32, lr: float = 1e-4):
        """Start fresh training or resume if already training (to update LR)."""
        if self.is_training:
            self.trainer.update_params({"lr": lr})
            return True
        
        self.is_training = True
        
        # Ensure trainer has latest LR
        if hasattr(self.trainer, 'update_params'):
            self.trainer.update_params({"lr": lr})
        
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
        return len(self.history) > 0

    def stop_training(self):
        if not self.is_training:
            return False
        
        self.trainer.request_stop()
        return True

    def reset_agent(self, agent_id: Optional[str] = None):
        """Stops training, deletes model, and resets agent weights."""
        if self.is_training:
            self.stop_training()
            if self.training_thread:
                self.training_thread.join(timeout=2.0)
        
        # If new agent_id provided, switch to it
        if agent_id:
            self.agent_id = agent_id
            self.model_path = f"models/{agent_id}_agent.pt"
        
        # Reset agent and trainer
        self.agent = registry.create(self.agent_id)
        self.trainer = self._create_trainer()
        
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
