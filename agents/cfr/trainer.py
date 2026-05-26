"""
CFR Trainer — integrates the CFR solver with the training infrastructure.

Overrides BaseTrainer.train() since CFR's iteration-based training is
fundamentally different from episode-based RL. The callback protocol
and evaluate() method are reused unchanged.

Metric mapping:
    "loss" → exploitability (both decrease during training)
    "avg_chips_per_round" → evaluation against HeuristicAgent
"""

import os
from typing import Dict, Optional, Callable

from preliminary_experiments.promoted_registry.base_trainer import BaseTrainer

from .solver import LeducCFRSolver


class CFRTrainer(BaseTrainer):
    """Trainer that runs CFR+ iterations on the agent's strategy store."""

    def __init__(self, agent, eval_interval: int = 200,
                 eval_num_games: int = 100, eval_opponent_factory=None):
        super().__init__(agent, eval_interval, eval_num_games, eval_opponent_factory)
        self.solver = LeducCFRSolver(agent.strategy_store)

    def train(self, num_episodes: int, batch_size: int = 32, save_path: str = None,
              callback: Optional[Callable] = None, start_episode: int = 0):
        """CFR training loop. Each 'episode' is one CFR+ iteration."""
        self.stop_requested = False

        for i in range(num_episodes):
            if self.stop_requested:
                print("Training stop requested.")
                break

            iteration = start_episode + i + 1
            self.solver.run_iteration(iteration)

            if (i + 1) % batch_size == 0:
                exploitability = self.solver.compute_exploitability()
                if callback:
                    callback({
                        "type": "batch_update",
                        "episode": iteration,
                        "loss": exploitability,
                    })
                if i < batch_size or iteration % 500 == 0:
                    print(f"Iteration {iteration}, Exploitability: {exploitability:.6f}")

            if iteration % self.eval_interval == 0:
                avg_chips = self.evaluate()
                if callback:
                    callback({
                        "type": "evaluation",
                        "episode": iteration,
                        "avg_chips_per_round": avg_chips,
                    })
                print(f"Iteration {iteration}, Avg Chips/Round: {avg_chips:+.2f}")

        if save_path:
            os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
            self.agent.save_model(save_path)
            print(f"Model saved to {save_path}")

    # Abstract stubs — never called since train() is overridden
    def collect_episode(self):
        raise NotImplementedError("CFR does not use episode collection")

    def update_model(self, batch_data: list) -> float:
        raise NotImplementedError("CFR does not use batch updates")

    def update_params(self, params: Dict):
        pass  # CFR has no learning rate or tunable params
