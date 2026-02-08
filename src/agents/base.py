import abc
from src.engine.leduc_game import Action
from src.engine.observation import Observation

class BaseAgent(abc.ABC):
    """
    Abstract base class for all Leduc Hold'em agents.
    Provides a consistent interface for the game engine.
    """

    @abc.abstractmethod
    def encode_observation(self, obs: Observation):
        """
        Transforms the game engine's Observation object into a format
        the agent's internal model can process (e.g., a torch.Tensor).
        
        Args:
            obs (Observation): Observation object from LeducGame.
            
        Returns:
            Any: Agent-specific representation of the game state.
        """
        pass

    @abc.abstractmethod
    def select_action(self, obs: Observation):
        """
        Takes an Observation and returns a game Action.
        
        Args:
            obs (Observation): Observation object from LeducGame.
            
        Returns:
            Action: The selected action.
        """
        pass

    def format_action(self, agent_output, legal_actions):
        """
        Maps the internal model output to a valid game Action.
        
        Args:
            agent_output (Any): Raw output from the agent's model.
            legal_actions (list): List of legal Action enums.
            
        Returns:
            Action: A valid game Action.
        """
        # Default implementation assumes agent_output is an index into Action enum
        # This can be overridden for more complex mapping logic.
        action = Action(agent_output)
        if action not in legal_actions:
            # Fallback strategy if model suggests illegal action
            return legal_actions[0] 
        return action
