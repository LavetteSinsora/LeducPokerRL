import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Tuple
from src.engine.leduc_game import LeducGame, Action
from src.agents.base import BaseAgent
from src.training.value_based_trainer import SelfPlayTrainer


class TDVariantTrainer(SelfPlayTrainer):
    """
    Trainer for systematic TD variant comparison.

    Extends SelfPlayTrainer with configurable n-step returns:
      - n_steps=1: TD(0) -- bootstrap from immediate next state
      - n_steps=2,3,...: n-step TD -- bootstrap from n steps ahead
      - n_steps=9999: Monte Carlo -- always use terminal reward (since
        Leduc chains are at most ~4 steps, 9999 effectively means infinity)

    The update rule for each time step t in a player's chain of length L:
      - If t + n_steps >= L: target = terminal reward (no bootstrapping)
      - Otherwise: target = V(s_{t+n_steps}) (bootstrap from future state)
    """

    def __init__(self, agent: BaseAgent, learning_rate: float = 1e-4,
                 n_steps: int = 1):
        super().__init__(agent, learning_rate=learning_rate)
        self.n_steps = n_steps

    def update_model(self, batch_data: list) -> float:
        """N-step TD update on per-player post-action state chains.

        Each chain: V(s_t) targets either V(s_{t+n}) (bootstrap) or
        terminal reward (when t+n >= chain length).
        """
        self.optimizer.zero_grad()
        total_losses = []

        for chains, rewards in batch_data:
            for p_idx in [0, 1]:
                chain = chains[p_idx]
                if not chain:
                    continue

                L = len(chain)
                for t in range(L):
                    prediction = self.agent.model(chain[t]).squeeze(0)

                    if t + self.n_steps >= L:
                        # Terminal or beyond horizon -- use actual reward
                        target = torch.FloatTensor([rewards[p_idx]])
                    else:
                        # Bootstrap from n-step-ahead state
                        with torch.no_grad():
                            target = self.agent.model(
                                chain[t + self.n_steps]
                            ).squeeze(0)

                    loss = self.criterion(prediction, target)
                    total_losses.append(loss)

        if total_losses:
            mean_loss = torch.stack(total_losses).mean()
            mean_loss.backward()
            self.optimizer.step()
            return mean_loss.item()
        return 0.0

    def update_params(self, params: Dict):
        """Update trainer parameters (learning rate and/or n_steps)."""
        if "lr" in params:
            new_lr = params["lr"]
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = new_lr
            print(f"Learning rate updated to: {new_lr}")

        if "n_steps" in params:
            self.n_steps = int(params["n_steps"])
            print(f"n_steps updated to: {self.n_steps}")
