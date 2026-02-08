"""
Evaluation Module for Leduc Hold'em Agents

This module provides a flexible, agent-agnostic evaluation framework. 
It is decoupled from the training infrastructure and can evaluate any 
two agents that implement the BaseAgent interface.

Key Design Decisions:
- Uses "average chips per round" as the primary metric (more informative than win rate)
- Supports any agent pair, not just trained agent vs. heuristic
- Returns detailed statistics for analysis
"""

from dataclasses import dataclass
from typing import List, Tuple
import random

from src.engine.leduc_game import LeducGame
from src.agents.base import BaseAgent


@dataclass
class EvaluationResult:
    """
    Contains the results of an evaluation run.
    
    Attributes:
        num_rounds: Total number of rounds played
        agent_0_avg_chips: Average chips won/lost per round by agent 0
        agent_1_avg_chips: Average chips won/lost per round by agent 1
        agent_0_total_chips: Total chips won/lost by agent 0
        agent_1_total_chips: Total chips won/lost by agent 1
        round_results: List of (agent_0_reward, agent_1_reward) for each round
    """
    num_rounds: int
    agent_0_avg_chips: float
    agent_1_avg_chips: float
    agent_0_total_chips: float
    agent_1_total_chips: float
    round_results: List[Tuple[float, float]]
    
    def __repr__(self) -> str:
        return (
            f"EvaluationResult(\n"
            f"  rounds={self.num_rounds},\n"
            f"  agent_0: {self.agent_0_avg_chips:+.2f} chips/round (total: {self.agent_0_total_chips:+.1f}),\n"
            f"  agent_1: {self.agent_1_avg_chips:+.2f} chips/round (total: {self.agent_1_total_chips:+.1f})\n"
            f")"
        )


def evaluate_agents(
    agent_0: BaseAgent,
    agent_1: BaseAgent,
    num_rounds: int = 100,
    randomize_positions: bool = True,
    verbose: bool = False
) -> EvaluationResult:
    """
    Evaluates two agents against each other over a number of rounds.
    
    This function is agent-agnostic and works with any implementation of BaseAgent.
    The primary metric is average chips per round, which provides more granular
    feedback than simple win rate.
    
    Args:
        agent_0: The first agent to evaluate
        agent_1: The second agent to evaluate
        num_rounds: Number of rounds to play (default: 100)
        randomize_positions: If True, randomly assign which agent plays as Player 0/1
                           each round. If False, agent_0 always plays as Player 0.
        verbose: If True, print progress during evaluation
        
    Returns:
        EvaluationResult: Detailed statistics about the evaluation
        
    Example:
        >>> from src.agents.value_based import ValueBasedAgent
        >>> from src.agents.heuristic import HeuristicAgent
        >>> 
        >>> trained_agent = ValueBasedAgent()
        >>> trained_agent.load_model("models/value_agent.pt")
        >>> opponent = HeuristicAgent()
        >>> 
        >>> result = evaluate_agents(trained_agent, opponent, num_rounds=100)
        >>> print(f"Trained agent: {result.agent_0_avg_chips:+.2f} chips/round")
    """
    game = LeducGame()
    round_results: List[Tuple[float, float]] = []
    
    # Track total chips for each agent (not player position)
    agent_0_total = 0.0
    agent_1_total = 0.0
    
    for round_num in range(num_rounds):
        game.reset()
        
        # Optionally randomize which agent plays which position
        if randomize_positions:
            swap_positions = random.choice([True, False])
        else:
            swap_positions = False
        
        # Map player ID to agent
        # Player 0 = agents[0], Player 1 = agents[1]
        if swap_positions:
            agents = [agent_1, agent_0]  # agent_1 plays as Player 0
        else:
            agents = [agent_0, agent_1]  # agent_0 plays as Player 0
        
        # Play one round
        while not game.is_finished:
            current_player = game.current_player
            obs = game.get_observation(viewer_id=current_player)
            
            current_agent = agents[current_player]
            action = current_agent.select_action(obs)
            
            game.step(action)
        
        # Get rewards (indexed by player position)
        rewards = game.get_reward()
        
        # Map back to agent rewards
        if swap_positions:
            agent_0_reward = rewards[1]  # agent_0 was Player 1
            agent_1_reward = rewards[0]  # agent_1 was Player 0
        else:
            agent_0_reward = rewards[0]  # agent_0 was Player 0
            agent_1_reward = rewards[1]  # agent_1 was Player 1
        
        agent_0_total += agent_0_reward
        agent_1_total += agent_1_reward
        round_results.append((agent_0_reward, agent_1_reward))
        
        if verbose and (round_num + 1) % 20 == 0:
            print(f"Round {round_num + 1}/{num_rounds}: "
                  f"Agent 0: {agent_0_total:+.1f}, Agent 1: {agent_1_total:+.1f}")
    
    return EvaluationResult(
        num_rounds=num_rounds,
        agent_0_avg_chips=agent_0_total / num_rounds,
        agent_1_avg_chips=agent_1_total / num_rounds,
        agent_0_total_chips=agent_0_total,
        agent_1_total_chips=agent_1_total,
        round_results=round_results
    )


def quick_evaluate(
    agent: BaseAgent,
    opponent: BaseAgent,
    num_rounds: int = 100
) -> float:
    """
    A simplified evaluation that returns just the average chips per round for the first agent.
    
    This is a convenience function for training loops that just need a single metric.
    
    Args:
        agent: The agent being evaluated
        opponent: The opponent agent
        num_rounds: Number of rounds to play
        
    Returns:
        float: Average chips per round for the agent being evaluated
    """
    result = evaluate_agents(agent, opponent, num_rounds=num_rounds, randomize_positions=True)
    return result.agent_0_avg_chips


if __name__ == "__main__":
    # Example usage / smoke test
    from src.agents.heuristic import HeuristicAgent
    from src.agents.value_based import ValueBasedAgent
    
    print("Evaluating two HeuristicAgents against each other...")
    agent_a = HeuristicAgent()
    agent_b = HeuristicAgent()
    
    result = evaluate_agents(agent_a, agent_b, num_rounds=100, verbose=True)
    print(f"\nResult: {result}")
    print(f"\nExpected: ~0 chips/round for both (symmetric matchup)")
    
    print("\n" + "="*50)
    print("Evaluating ValueBasedAgent (untrained) vs HeuristicAgent...")
    
    value_agent = ValueBasedAgent()
    result = evaluate_agents(value_agent, agent_b, num_rounds=100, verbose=True)
    print(f"\nResult: {result}")
