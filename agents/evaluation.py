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

from agents.base import BaseAgent
from engine.poker_session import PokerSession


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
    session_length: int = None,
    verbose: bool = False
) -> EvaluationResult:
    """
    Evaluates two agents against each other over a number of rounds.

    This function is agent-agnostic and works with any implementation of BaseAgent.
    The primary metric is average chips per round, which provides more granular
    feedback than simple win rate.

    Uses two PokerSession instances with fixed positions so opponent stats accumulate
    cleanly. The first half of rounds have agent_0 as player 0; the second half have
    agent_0 as player 1. Both positions are covered, and each session's stats only
    ever reflect one agent's actions (no position-swap confounding).

    Args:
        agent_0: The first agent to evaluate
        agent_1: The second agent to evaluate
        num_rounds: Number of rounds to play (default: 100; rounded down to nearest even)
        session_length: If set, reset PokerSession stats every N hands. Use this to
            match the training-time session length for stateful agents (e.g., 100 for
            adaptive_value / modulated_value). None = legacy continuous behavior.
        verbose: If True, print progress during evaluation

    Returns:
        EvaluationResult: Detailed statistics about the evaluation

    Example:
        >>> from agents.heuristic.agent import HeuristicAgent
        >>> from agents.value_based.agent import ValueBasedAgent
        >>> trained_agent = ValueBasedAgent()
        >>> opponent = HeuristicAgent()
        >>> result = evaluate_agents(trained_agent, opponent, num_rounds=100)
        >>> print(f"Trained agent: {result.agent_0_avg_chips:+.2f} chips/round")
    """
    # Two sessions with fixed positions so stats stay clean.
    # Session A: agent_0 as player 0, agent_1 as player 1 (first half).
    # Session B: agent_0 as player 1, agent_1 as player 0 (second half).
    # Each session's stats[position] only ever sees one agent's actions,
    # so opponent models are unconfounded. Together both positions are covered.
    half = num_rounds // 2

    round_results: List[Tuple[float, float]] = []
    agent_0_total = 0.0
    agent_1_total = 0.0

    for agents, a0_pos in [
        ([agent_0, agent_1], 0),  # agent_0 is player 0
        ([agent_1, agent_0], 1),  # agent_0 is player 1
    ]:
        session = PokerSession()
        hands_in_session = 0

        for _ in range(half):
            # Reset session when session_length is set and block is full.
            # This mirrors training-time distribution for stateful agents.
            if session_length is not None and hands_in_session >= session_length:
                session = PokerSession()
                hands_in_session = 0

            session.new_hand()

            while not session.is_finished:
                current_player = session.current_player
                obs = session.get_observation(viewer_id=current_player)
                action = agents[current_player].select_action(obs)
                session.step(action)

            hands_in_session += 1
            rewards = session.get_reward()
            agent_0_reward = rewards[a0_pos]
            agent_1_reward = rewards[1 - a0_pos]
            agent_0_total += agent_0_reward
            agent_1_total += agent_1_reward
            round_results.append((agent_0_reward, agent_1_reward))

            if verbose and len(round_results) % 20 == 0:
                print(f"Round {len(round_results)}/{num_rounds}: "
                      f"Agent 0: {agent_0_total:+.1f}, Agent 1: {agent_1_total:+.1f}")

    total_rounds = half * 2
    return EvaluationResult(
        num_rounds=total_rounds,
        agent_0_avg_chips=agent_0_total / total_rounds,
        agent_1_avg_chips=agent_1_total / total_rounds,
        agent_0_total_chips=agent_0_total,
        agent_1_total_chips=agent_1_total,
        round_results=round_results
    )


def compute_robustness_metrics(scores: dict) -> dict:
    """
    Compute multi-dimensional robustness metrics from head-to-head scores.

    Given a dict of {opponent_id: avg_chips_per_round}, returns:
      - avg: Mean performance across all opponents (overall strength)
      - worst_case: Minimum performance (how badly the worst opponent hurts us)
      - std: Standard deviation (consistency across opponent types)
      - robustness: avg - 1.5 * std (lower confidence bound at ~93.3%)

    The robustness score is a risk-adjusted metric inspired by portfolio theory.
    It answers: "What performance can we be 93.3% confident of exceeding?"
    In a normal distribution, avg - 1.5*std is the ~6.7th percentile.

    Args:
        scores: Dict mapping opponent_id to avg chips/round against that opponent.
                Self-matchups (score=0) should be excluded.

    Returns:
        Dict with keys: avg, worst_case, best_case, std, robustness, n_opponents
    """
    import math

    values = [v for v in scores.values() if v != 0.0 or len(scores) == 1]
    if not values:
        return {"avg": 0.0, "worst_case": 0.0, "best_case": 0.0,
                "std": 0.0, "robustness": 0.0, "n_opponents": 0}

    n = len(values)
    avg = sum(values) / n
    worst = min(values)
    best = max(values)

    if n > 1:
        variance = sum((v - avg) ** 2 for v in values) / (n - 1)  # sample std
        std = math.sqrt(variance)
    else:
        std = 0.0

    robustness = avg - 1.5 * std

    return {
        "avg": round(avg, 4),
        "worst_case": round(worst, 4),
        "best_case": round(best, 4),
        "std": round(std, 4),
        "robustness": round(robustness, 4),
        "n_opponents": n,
    }


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
    result = evaluate_agents(agent, opponent, num_rounds=num_rounds)
    return result.agent_0_avg_chips


if __name__ == "__main__":
    # Example usage / smoke test
    from agents.heuristic.agent import HeuristicAgent
    from agents.value_based.agent import ValueBasedAgent
    
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
