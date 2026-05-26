"""
PoolTrainer — TD(0) training against a rotating pool of rule-based opponents.

The learner (ValueBasedAgent) trains as a single player alternating positions
each episode. The opponent is drawn from a fixed pool and rotated on a
schedule (round-robin, every `rotate_every` episodes).

This is the core difference from SelfPlayTrainer: the opponent is external
and stationary, not a co-evolving copy of the learner.
"""

import os
import sys

import torch
import torch.nn as nn
import torch.optim as optim
from typing import List, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from agents.base import BaseAgent
from preliminary_experiments.promoted_registry.base_trainer import BaseTrainer
from engine.leduc_game import LeducGame, Action


class PoolTrainer(BaseTrainer):
    """
    Trains a ValueBasedAgent via TD(0) against a rotating pool of opponents.

    Episode structure:
      - Learner alternates positions: even episodes → P0, odd → P1
      - Only the learner's post-action chain is recorded and trained on
      - Opponent is selected round-robin from the pool, rotated every
        `rotate_every` episodes

    Args:
        agent:        The ValueBasedAgent to train.
        learning_rate: Adam learning rate.
        rotate_every:  Episodes between opponent rotations (default 1000).
    """

    def __init__(self, agent: BaseAgent, learning_rate: float = 1e-4,
                 rotate_every: int = 1000):
        super().__init__(agent, eval_interval=50, eval_num_games=100)
        self.optimizer = optim.Adam(agent.model.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()
        self.game = LeducGame()
        self.rotate_every = rotate_every
        self.episode_count = 0
        self.current_opponent_idx = 0
        self.opponent_pool = self._build_pool()

    # ------------------------------------------------------------------
    # Pool construction
    # ------------------------------------------------------------------

    def _build_pool(self) -> List[Tuple[str, BaseAgent]]:
        """Instantiate and return the ordered opponent pool."""
        from agents.rule_based.random_agent     import RandomAgent
        from agents.rule_based.maniac           import ManiacAgent
        from agents.rule_based.tight_passive    import TightPassiveAgent
        from agents.rule_based.loose_passive    import LoosePassiveAgent
        from agents.rule_based.loose_aggressive import LooseAggressiveAgent
        from agents.rule_based.tight_aggressive import TightAggressiveAgent
        from agents.heuristic.agent import HeuristicAgent
        from agents.cfr.agent import CFRAgent

        cfr_path = os.path.join(ROOT, "agents", "cfr", "checkpoint.pt")
        pool = [
            ("random",           RandomAgent()),
            ("maniac",           ManiacAgent()),
            ("tight_passive",    TightPassiveAgent()),
            ("loose_passive",    LoosePassiveAgent()),
            ("loose_aggressive", LooseAggressiveAgent()),
            ("tight_aggressive", TightAggressiveAgent()),
            ("heuristic",        HeuristicAgent()),
            ("cfr",              CFRAgent(model_path=cfr_path)),
        ]
        for _, opp in pool:
            opp.set_train_mode(False)
        print(f"[PoolTrainer] Pool: {[name for name, _ in pool]}")
        return pool

    # ------------------------------------------------------------------
    # BaseTrainer hooks
    # ------------------------------------------------------------------

    def collect_episode(self) -> Tuple[List[torch.Tensor], float]:
        """
        Play one hand. Only the learner's post-action states are recorded.

        Returns:
            learner_chain: list of encoded post-action tensors for the learner
            learner_reward: scalar chip result for the learner
        """
        self.game.reset()
        learner_id = self.episode_count % 2   # alternate P0 / P1 each episode
        opponent = self.opponent_pool[self.current_opponent_idx][1]
        learner_chain: List[torch.Tensor] = []

        while not self.game.is_finished:
            cp = self.game.current_player
            obs = self.game.get_observation(viewer_id=cp)

            if cp == learner_id:
                action = self.agent.select_action(obs)
                if isinstance(action, tuple):
                    action = action[0]
                # Record post-action state (with board masking via simulate_action)
                post_obs, _ = LeducGame.simulate_action(obs, action)
                encoded = self.agent.encode_observation(post_obs, viewer_id=learner_id)
                learner_chain.append(encoded)
            else:
                action = opponent.select_action(obs)
                if isinstance(action, tuple):
                    action = action[0]

            self.game.step(action)

        rewards = self.game.get_reward()
        learner_reward = rewards[learner_id]

        self.episode_count += 1
        self._maybe_rotate()

        return learner_chain, learner_reward

    def update_model(self, batch_data: list) -> float:
        """
        TD(0) update on the learner's chain.

        For each episode in the batch:
          - V(s_t) → V(s_{t+1})  for all t < last
          - V(s_last) → terminal reward

        Returns:
            mean TD MSE loss over the batch.
        """
        self.optimizer.zero_grad()
        losses = []

        for learner_chain, learner_reward in batch_data:
            if not learner_chain:
                continue
            for t in range(len(learner_chain)):
                pred = self.agent.model(learner_chain[t]).squeeze(0)
                if t == len(learner_chain) - 1:
                    target = torch.FloatTensor([learner_reward])
                else:
                    with torch.no_grad():
                        target = self.agent.model(learner_chain[t + 1]).squeeze(0)
                losses.append(self.criterion(pred, target))

        if losses:
            mean_loss = torch.stack(losses).mean()
            mean_loss.backward()
            self.optimizer.step()
            return mean_loss.item()
        return 0.0

    def update_params(self, params: dict):
        if "lr" in params:
            for pg in self.optimizer.param_groups:
                pg["lr"] = params["lr"]
        if "rotate_every" in params:
            self.rotate_every = params["rotate_every"]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _maybe_rotate(self):
        """Advance to the next opponent after every `rotate_every` episodes."""
        if self.episode_count > 0 and self.episode_count % self.rotate_every == 0:
            self.current_opponent_idx = (
                (self.current_opponent_idx + 1) % len(self.opponent_pool)
            )
            name = self.opponent_pool[self.current_opponent_idx][0]
            print(f"  [ep={self.episode_count:,}] rotating → {name}")

    @property
    def current_opponent_name(self) -> str:
        return self.opponent_pool[self.current_opponent_idx][0]
