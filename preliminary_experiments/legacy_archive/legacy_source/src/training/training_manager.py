import json
import threading
import os
from typing import List, Dict, Optional
from src.agents.registry import registry

LOSS_TYPE_MAP = {
    "PolicyGradientTrainer": "REINFORCE Loss",
    "ActorCriticTrainer": "Policy Gradient Loss",
    "EntropyACTrainer": "Policy Gradient Loss",
    "InfoHidingTrainer": "Policy Gradient Loss",
    "CFRTrainer": "Regret",
}


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
        self.history_path = self.model_path.replace('.pt', '_history.json')

        # Metrics storage — persisted to disk so charts survive server restarts
        self.history: List[Dict] = []
        self.current_stats = {
            "episode": 0,
            "loss": 0.0,
            "avg_chips_per_round": 0.0
        }
        self._load_history()

        # Matchup evaluation: agent IDs to evaluate against at each eval interval
        self.matchup_opponents: List[str] = []

    def set_matchup_opponents(self, opponent_ids: List[str]):
        """Update the list of opponents for matchup evaluation.

        Safe to call while training is running — the next eval interval
        will pick up the change.
        """
        self.matchup_opponents = [oid for oid in opponent_ids if oid != self.agent_id]

    def _load_history(self):
        """Load training history from disk if available."""
        if os.path.exists(self.history_path):
            try:
                with open(self.history_path, 'r') as f:
                    self.history = json.load(f)
                # Restore episode counter and latest stats from history
                if self.history:
                    last_episode = max(h.get("episode", 0) for h in self.history)
                    self.current_stats["episode"] = last_episode
                    # Restore last loss and chips values
                    for h in reversed(self.history):
                        if h["type"] == "loss" and self.current_stats["loss"] == 0.0:
                            self.current_stats["loss"] = h["loss"]
                        if h["type"] == "avg_chips" and self.current_stats["avg_chips_per_round"] == 0.0:
                            self.current_stats["avg_chips_per_round"] = h["avg_chips_per_round"]
                        if self.current_stats["loss"] != 0.0 and self.current_stats["avg_chips_per_round"] != 0.0:
                            break
                print(f"Loaded training history ({len(self.history)} entries, episode {self.current_stats['episode']})")
            except Exception as e:
                print(f"Error loading history: {e}")

    def _save_history(self):
        """Persist training history to disk."""
        try:
            os.makedirs(os.path.dirname(self.history_path) or '.', exist_ok=True)
            with open(self.history_path, 'w') as f:
                json.dump(self.history, f)
        except Exception as e:
            print(f"Error saving history: {e}")

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
            self._save_history()
        elif data["type"] == "evaluation":
            self.current_stats["episode"] = data["episode"]
            self.current_stats["avg_chips_per_round"] = data["avg_chips_per_round"]
            self.history.append({
                "episode": data["episode"],
                "avg_chips_per_round": data["avg_chips_per_round"],
                "type": "avg_chips"
            })

            # Run matchup evaluations against selected opponents
            if self.matchup_opponents:
                from src.training.evaluation import quick_evaluate
                self.agent.set_train_mode(False)
                for opp_id in self.matchup_opponents:
                    try:
                        opponent = registry.create(opp_id)
                        model_path = f"models/{opp_id}_agent.pt"
                        if os.path.exists(model_path):
                            opponent.load_model(model_path)
                        avg_chips = quick_evaluate(self.agent, opponent, num_rounds=100)
                        self.history.append({
                            "episode": data["episode"],
                            "opponent_id": opp_id,
                            "avg_chips_per_round": avg_chips,
                            "type": "matchup"
                        })
                    except Exception as e:
                        print(f"Matchup eval error vs {opp_id}: {e}")
                self.agent.set_train_mode(True)

            self._save_history()

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
            self.history_path = self.model_path.replace('.pt', '_history.json')
        
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
        self.matchup_opponents = []
        
        # Delete model and history files if they exist
        for path in [self.model_path, self.history_path]:
            if os.path.exists(path):
                os.remove(path)
                print(f"Deleted: {path}")
        
        return True

    def get_status(self):
        trainer_name = type(self.trainer).__name__
        loss_type = LOSS_TYPE_MAP.get(trainer_name, "MSE Loss")
        return {
            "is_training": self.is_training,
            "has_history": self.has_training_history,
            "stats": self.current_stats,
            "loss_type": loss_type,
            "matchup_opponents": self.matchup_opponents
        }

    def get_history(self):
        return self.history

    def run_debug_episode(self) -> Dict:
        """Runs a single episode in debug mode and returns the trace."""
        return self.trainer.debug_episode()
