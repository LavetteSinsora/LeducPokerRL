"""
Belief Oracle Trainer -- Self-play trainer for BeliefOracleAgent.

Training protocol:
  1. Self-play using the belief oracle agent against itself.
  2. Both hands are known during self-play.
  3. The VALUE network is trained on ground truth: V(state, my_hand, opp_hand)
     using the ACTUAL opponent hand (perfect information).
  4. The POLICY (action selection) uses BELIEF-WEIGHTED values, same as
     during evaluation -- no distribution shift.

Supports two training methods:
  - Monte Carlo (primary): target = terminal reward for ALL states in the chain.
    Games are 1-3 steps per player, so MC has very low variance.
  - TD(0) (comparison): target = reward + gamma * V(next_state, my_hand, opp_hand)
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, List, Tuple, Optional
from src.engine.leduc_game import LeducGame, Action
from src.engine.observation import Observation
from src.agents.belief_oracle import BeliefOracleAgent
from src.training.base import BaseTrainer


class BeliefOracleTrainer(BaseTrainer):
    """
    Trainer for BeliefOracleAgent.

    Key design: the value function is trained with ground-truth opponent hand
    (perfect info), but action selection during training uses belief-weighted
    values (imperfect info), matching evaluation exactly.

    Supports both Monte Carlo and TD(0) training methods.
    """

    def __init__(self, agent: BeliefOracleAgent, learning_rate: float = 1e-4,
                 method: str = 'mc', gamma: float = 1.0):
        """
        Args:
            agent: The BeliefOracleAgent to train.
            learning_rate: Learning rate for the value network.
            method: Training method -- 'mc' (Monte Carlo) or 'td' (TD(0)).
            gamma: Discount factor (default 1.0 for episodic games).
        """
        super().__init__(agent, eval_interval=50, eval_num_games=100)

        self.optimizer = optim.Adam(
            self.agent.model.parameters(), lr=learning_rate
        )
        self.criterion = nn.MSELoss()
        self.game = LeducGame()
        self.method = method
        self.gamma = gamma

    def collect_episode(self) -> Tuple[List[List[dict]], List[float]]:
        """
        Play one episode of self-play.

        Both hands are known. Action selection uses belief-weighted values
        (via the agent's select_action), but training data records the
        TRUE opponent hand for each state.

        Returns:
            chains: per-player lists of state dicts containing:
                - 'encoded': tensor encoding WITH true opp hand (for training)
                - 'viewer_id': which player's perspective
            rewards: final rewards for each player
        """
        self.game.reset()
        chains = [[], []]  # chains[p] = list of dicts for player p

        while not self.game.is_finished:
            current_player = self.game.current_player
            opponent = 1 - current_player
            obs = self.game.get_observation(viewer_id=current_player)

            # Action selection uses BELIEF-WEIGHTED values (matches evaluation)
            action = self.agent.select_action(obs)
            if isinstance(action, tuple):
                action = action[0]

            # Record post-action state WITH TRUE opponent hand for training
            post_obs, _ = LeducGame.simulate_action(obs, action)

            # Get the TRUE opponent hand index
            opp_hand = self.game.player_hands[opponent]
            opp_hand_idx = self.agent.CARD_MAP[opp_hand]

            # Encode with true opponent hand
            encoded = self.agent.encode_state_with_opp(
                post_obs, viewer_id=current_player, opp_hand_idx=opp_hand_idx
            )

            chains[current_player].append({
                'encoded': encoded,
                'viewer_id': current_player,
            })

            self.game.step(action)

        rewards = self.game.get_reward()
        return chains, rewards

    def update_model(self, batch_data: list) -> float:
        """
        Update the value network using collected episode data.

        Dispatches to MC or TD(0) based on self.method.

        Args:
            batch_data: List of (chains, rewards) tuples.

        Returns:
            Mean loss value.
        """
        if self.method == 'mc':
            return self._update_mc(batch_data)
        else:
            return self._update_td(batch_data)

    def _update_mc(self, batch_data: list) -> float:
        """
        Monte Carlo update: target = terminal reward for ALL states.

        Since games are 1-3 steps per player, MC has very low variance.
        No bootstrapping -> no TD coupling issues -> no stale target problem.
        """
        self.optimizer.zero_grad()
        total_losses = []

        for chains, rewards in batch_data:
            for p_idx in [0, 1]:
                chain = chains[p_idx]
                if not chain:
                    continue

                terminal_reward = rewards[p_idx]
                target = torch.FloatTensor([terminal_reward])

                for entry in chain:
                    prediction = self.agent.model(entry['encoded']).squeeze(0)
                    loss = self.criterion(prediction, target)
                    total_losses.append(loss)

        if total_losses:
            mean_loss = torch.stack(total_losses).mean()
            mean_loss.backward()
            self.optimizer.step()
            return mean_loss.item()
        return 0.0

    def _update_td(self, batch_data: list) -> float:
        """
        TD(0) update: target = reward + gamma * V(next_state, my_hand, opp_hand).

        The opponent hand is CONSTANT within a game, so the TD target uses
        the same opp_hand encoding throughout.
        """
        self.optimizer.zero_grad()
        total_losses = []

        for chains, rewards in batch_data:
            for p_idx in [0, 1]:
                chain = chains[p_idx]
                if not chain:
                    continue

                for t in range(len(chain)):
                    prediction = self.agent.model(chain[t]['encoded']).squeeze(0)

                    if t == len(chain) - 1:
                        # Last action -> target is terminal reward
                        target = torch.FloatTensor([rewards[p_idx]])
                    else:
                        # Bootstrap from next state (same player's chain)
                        with torch.no_grad():
                            next_val = self.agent.model(
                                chain[t + 1]['encoded']
                            ).squeeze(0)
                        target = self.gamma * next_val

                    loss = self.criterion(prediction, target)
                    total_losses.append(loss)

        if total_losses:
            mean_loss = torch.stack(total_losses).mean()
            mean_loss.backward()
            self.optimizer.step()
            return mean_loss.item()
        return 0.0

    def debug_episode(self) -> Dict:
        """Play one episode and record detailed trace with belief vectors."""
        self.game.reset()
        episode_trace = []

        old_train_mode = self.agent.train_mode
        self.agent.set_train_mode(False)

        while not self.game.is_finished:
            current_player = self.game.current_player
            obs = self.game.get_observation(viewer_id=current_player)
            belief = self.agent.compute_belief_from_history(obs)

            evaluations = self.agent.get_action_evaluations(obs)
            selected_eval = max(evaluations, key=lambda x: x["value"])
            action = selected_eval["action"]

            step_info = {
                "player_id": current_player,
                "observation": {
                    "player_hand": obs.player_hand,
                    "board": obs.board,
                    "pot": obs.pot,
                    "current_round": obs.current_round,
                },
                "true_opp_hand": self.game.player_hands[1 - current_player],
                "belief": belief.tolist(),
                "evaluations": [
                    {
                        "action": e["action"].name,
                        "value": e["value"],
                        "belief": e["belief"].tolist(),
                    } for e in evaluations
                ],
                "selected_action": action.name,
            }
            episode_trace.append(step_info)
            self.game.step(action)

        rewards = self.game.get_reward()

        for step in episode_trace:
            step["true_value"] = rewards[step["player_id"]]

        self.agent.set_train_mode(old_train_mode)
        return {
            "trace": episode_trace,
            "final_rewards": rewards,
            "eval_type": "belief_oracle",
            "player_hands": list(self.game.player_hands),
        }

    def update_params(self, params: Dict):
        """Update training parameters."""
        if "lr" in params:
            for pg in self.optimizer.param_groups:
                pg['lr'] = params["lr"]
        if "method" in params:
            self.method = params["method"]
