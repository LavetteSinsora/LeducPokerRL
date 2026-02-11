import abc
from typing import Any, List
from src.engine.leduc_game import Action
from src.engine.observation import Observation

class BaseAgent(abc.ABC):
    """
    Abstract base class for all Leduc Hold'em agents.

    Required to implement:
        select_action(obs) -> Action

    Optional overrides (have sensible defaults):
        encode_observation(obs, **kwargs) -> Any   (default: returns obs as-is)
        get_action_evaluations(obs) -> list         (default: returns [])
        save_model(path) -> None                    (default: no-op)
        load_model(path) -> None                    (default: no-op)
        set_train_mode(mode) -> None                (default: no-op)
    """

    @abc.abstractmethod
    def select_action(self, obs: Observation) -> Action:
        """
        Takes an Observation and returns a game Action.

        Args:
            obs: Observation object from LeducGame.

        Returns:
            The selected Action.
        """
        pass

    def encode_observation(self, obs: Observation, **kwargs) -> Any:
        """
        Transforms the Observation into a format the agent can process.
        Rule-based agents can leave the default (returns obs unchanged).

        Args:
            obs: Observation object from LeducGame.

        Returns:
            Agent-specific representation of the game state.
        """
        return obs

    def get_action_evaluations(self, obs: Observation) -> list:
        """
        Returns per-action value estimates for the analyzer UI.
        Agents without value estimation return an empty list,
        which the analyzer renders gracefully as "no data".
        """
        return []

    def save_model(self, path: str) -> None:
        """Persist model weights to *path*. No-op for non-trainable agents."""
        pass

    def load_model(self, path: str) -> None:
        """Load model weights from *path*. No-op for non-trainable agents."""
        pass

    def set_train_mode(self, mode: bool) -> None:
        """Toggle train/eval mode. No-op for non-trainable agents."""
        pass
