"""
Agent Registry - Single source of truth for all available agents.

This module provides a centralized registry for all agents in the system.
New agents should be registered here to be discoverable by the web UI,
training systems, and other components.

Usage:
    from src.agents.registry import registry

    # List all available agents
    agents = registry.list_agents()

    # Create an agent instance
    agent = registry.create("heuristic")

    # Create with custom kwargs
    agent = registry.create("value_based", model_path="path/to/model.pt")
"""

from dataclasses import dataclass, field
from typing import Type, Dict, Any, Optional, List


@dataclass
class AgentMetadata:
    """Metadata describing an agent type."""
    id: str                          # Unique identifier (e.g., "heuristic", "value_based")
    display_name: str                # Human-readable name (e.g., "Heuristic Agent")
    description: str                 # Description for UI tooltips
    is_trainable: bool = False       # Whether this agent can be trained
    requires_model_path: bool = False  # Whether agent needs a model file
    category: str = "general"        # Category for grouping (e.g., "rule_based", "rl")
    trainer_class: Optional[Type] = None  # Trainer class for trainable agents


class AgentRegistry:
    """
    Central registry for all agent types.
    
    This class maintains a mapping of agent IDs to their classes and metadata,
    enabling dynamic discovery and instantiation of agents throughout the system.
    """
    
    def __init__(self):
        self._agents: Dict[str, Type] = {}
        self._metadata: Dict[str, AgentMetadata] = {}
        self._factory_kwargs: Dict[str, Dict[str, Any]] = {}
    
    def register(
        self, 
        id: str, 
        agent_class: Type, 
        metadata: AgentMetadata,
        default_kwargs: Optional[Dict[str, Any]] = None
    ):
        """
        Register an agent type with the registry.
        
        Args:
            id: Unique identifier for the agent type
            agent_class: The agent class to instantiate
            metadata: AgentMetadata describing the agent
            default_kwargs: Default keyword arguments for instantiation
        """
        self._agents[id] = agent_class
        self._metadata[id] = metadata
        self._factory_kwargs[id] = default_kwargs or {}
    
    def list_agents(self, category: Optional[str] = None) -> List[AgentMetadata]:
        """
        List all registered agents, optionally filtered by category.
        
        Args:
            category: Optional category to filter by
            
        Returns:
            List of AgentMetadata for matching agents
        """
        agents = list(self._metadata.values())
        if category:
            agents = [a for a in agents if a.category == category]
        return agents
    
    def get_metadata(self, agent_id: str) -> Optional[AgentMetadata]:
        """
        Get metadata for a specific agent.
        
        Args:
            agent_id: The agent's unique identifier
            
        Returns:
            AgentMetadata if found, None otherwise
        """
        return self._metadata.get(agent_id)
    
    def create(self, agent_id: str, **kwargs):
        """
        Create an instance of an agent by its ID.
        
        Args:
            agent_id: The agent's unique identifier
            **kwargs: Additional keyword arguments to pass to the constructor
            
        Returns:
            An instance of the requested agent
            
        Raises:
            ValueError: If agent_id is not registered
        """
        if agent_id not in self._agents:
            raise ValueError(f"Unknown agent type: {agent_id}. "
                           f"Available: {list(self._agents.keys())}")
        
        # Merge default kwargs with provided kwargs (provided takes precedence)
        merged_kwargs = {**self._factory_kwargs[agent_id], **kwargs}
        return self._agents[agent_id](**merged_kwargs)
    
    def is_registered(self, agent_id: str) -> bool:
        """
        Check if an agent ID is registered.
        
        Args:
            agent_id: The agent's unique identifier
            
        Returns:
            True if registered, False otherwise
        """
        return agent_id in self._agents
    
    def get_trainable_agents(self) -> List[AgentMetadata]:
        """
        Get all agents that can be trained.
        
        Returns:
            List of AgentMetadata for trainable agents
        """
        return [a for a in self._metadata.values() if a.is_trainable]


# Global registry instance - single source of truth
registry = AgentRegistry()


# ============================================================
# Register all built-in agents
# ============================================================

def _register_builtin_agents():
    """Register all built-in agents with the registry."""
    # Import here to avoid circular imports
    from .heuristic import HeuristicAgent
    from .value_based import ValueBasedAgent
    from .policy_gradient import PolicyGradientAgent
    from .adaptive_value import AdaptiveValueAgent
    from .aux_value import AuxValueAgent
    from .cfr_agent import CFRAgent
    from .actor_critic import ActorCriticAgent
    from .history_value import HistoryValueAgent
    from .decay_adaptive import DecayAdaptiveAgent
    from .nstep_value import NStepValueAgent
    from src.training.value_based_trainer import SelfPlayTrainer
    from src.training.policy_gradient_trainer import PolicyGradientTrainer
    from src.training.adaptive_trainer import AdaptiveTrainer
    from src.training.aux_value_trainer import AuxValueTrainer
    from src.training.cfr_trainer import CFRTrainer
    from src.training.actor_critic_trainer import ActorCriticTrainer
    from src.training.history_value_trainer import HistoryValueTrainer
    from src.training.decay_adaptive_trainer import DecayAdaptiveTrainer
    from src.training.nstep_value_trainer import NStepValueTrainer

    # Heuristic Agent - rule-based baseline
    registry.register(
        id="heuristic",
        agent_class=HeuristicAgent,
        metadata=AgentMetadata(
            id="heuristic",
            display_name="Heuristic Agent",
            description="A rule-based agent using hand-crafted poker strategy",
            is_trainable=False,
            category="rule_based"
        )
    )

    # Value-Based RL Agent - trainable neural network
    registry.register(
        id="value_based",
        agent_class=ValueBasedAgent,
        metadata=AgentMetadata(
            id="value_based",
            display_name="Value Network AI",
            description="RL agent using a learned value function",
            is_trainable=True,
            requires_model_path=True,
            category="rl",
            trainer_class=SelfPlayTrainer
        )
    )

    # Policy Gradient RL Agent - trainable neural network
    registry.register(
        id="policy_gradient",
        agent_class=PolicyGradientAgent,
        metadata=AgentMetadata(
            id="policy_gradient",
            display_name="Policy Gradient AI",
            description="RL agent using policy gradient (REINFORCE) algorithm",
            is_trainable=True,
            requires_model_path=True,
            category="rl",
            trainer_class=PolicyGradientTrainer
        )
    )

    # Adaptive Value Agent - exploits opponent tendencies via session stats
    registry.register(
        id="adaptive_value",
        agent_class=AdaptiveValueAgent,
        metadata=AgentMetadata(
            id="adaptive_value",
            display_name="Adaptive Value AI",
            description="Exploits opponent tendencies using cross-hand behavior statistics",
            is_trainable=True,
            requires_model_path=True,
            category="rl",
            trainer_class=AdaptiveTrainer
        )
    )

    # Auxiliary Value Agent - pre-action Bellman consistency auxiliary loss
    registry.register(
        id="aux_value",
        agent_class=AuxValueAgent,
        metadata=AgentMetadata(
            id="aux_value",
            display_name="Aux Value AI",
            description="Value agent with pre-action Bellman consistency auxiliary loss",
            is_trainable=True,
            requires_model_path=True,
            category="rl",
            trainer_class=AuxValueTrainer
        )
    )

    # History Value Agent - action history augmented value network
    registry.register(
        id="history_value",
        agent_class=HistoryValueAgent,
        metadata=AgentMetadata(
            id="history_value",
            display_name="History Value AI",
            description="Value agent augmented with scalable intra-hand action history encoding",
            is_trainable=True,
            requires_model_path=True,
            category="rl",
            trainer_class=HistoryValueTrainer
        )
    )

    # Decay Adaptive Agent - EMA-weighted opponent stats
    registry.register(
        id="decay_adaptive",
        agent_class=DecayAdaptiveAgent,
        metadata=AgentMetadata(
            id="decay_adaptive",
            display_name="Decay Adaptive AI",
            description="Adaptive value agent with EMA-weighted opponent statistics for faster adaptation",
            is_trainable=True,
            requires_model_path=True,
            category="rl",
            trainer_class=DecayAdaptiveTrainer
        )
    )

    # CFR Nash Equilibrium Agent - game-theoretic baseline
    registry.register(
        id="cfr",
        agent_class=CFRAgent,
        metadata=AgentMetadata(
            id="cfr",
            display_name="CFR Nash Equilibrium",
            description="Game-theoretic baseline via Counterfactual Regret Minimization",
            is_trainable=True,
            requires_model_path=True,
            category="game_theory",
            trainer_class=CFRTrainer
        )
    )

    # Actor-Critic RL Agent - policy gradient with learned value baseline
    registry.register(
        id="actor_critic",
        agent_class=ActorCriticAgent,
        metadata=AgentMetadata(
            id="actor_critic",
            display_name="Actor-Critic AI",
            description="RL agent using REINFORCE with a learned value baseline for variance reduction",
            is_trainable=True,
            requires_model_path=True,
            category="rl",
            trainer_class=ActorCriticTrainer
        )
    )

    # N-Step Value Agent - value network with n-step return targets
    registry.register(
        id="nstep_value",
        agent_class=NStepValueAgent,
        metadata=AgentMetadata(
            id="nstep_value",
            display_name="N-Step Value AI",
            description="Value agent using n-step returns for less biased TD targets",
            is_trainable=True,
            requires_model_path=True,
            category="rl",
            trainer_class=NStepValueTrainer
        )
    )


# Initialize built-in agents on module import
_register_builtin_agents()
