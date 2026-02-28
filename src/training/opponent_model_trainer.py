"""
Opponent Model Trainer — Self-play trainer for OpponentModelAgent.

Trains two networks simultaneously:
  1. Value network: TD(0) on post-action state chains (same as SelfPlayTrainer)
  2. Opponent model: Cross-entropy loss on (state, opponent_action) pairs

During each game, whenever the OPPONENT acts, we record
(encoded_state_from_opponent_perspective, action_index) for the opponent model.

The 2-ply planning only happens at ACTION SELECTION time (not training time).
Training uses standard 1-ply chains for the value network.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Tuple
from src.engine.leduc_game import LeducGame, Action
from src.agents.base import BaseAgent
from src.training.base import BaseTrainer


class OpponentModelTrainer(BaseTrainer):
    """
    Self-play trainer that trains both value network and opponent model.

    The value network learns state values via TD(0) on post-action chains.
    The opponent model learns to predict opponent actions via cross-entropy.
    """

    def __init__(self, agent, learning_rate=1e-4):
        super().__init__(agent, eval_interval=50, eval_num_games=100)

        # Separate optimizers for value network and opponent model
        self.value_optimizer = optim.Adam(
            self.agent.model.parameters(), lr=learning_rate
        )
        self.opp_optimizer = optim.Adam(
            self.agent.opponent_model.parameters(), lr=learning_rate
        )
        self.value_criterion = nn.MSELoss()
        self.opp_criterion = nn.CrossEntropyLoss()
        self.game = LeducGame()

    def collect_episode(self) -> Tuple[List[List[torch.Tensor]], List[float],
                                       List[Tuple[torch.Tensor, int]]]:
        """
        Play one self-play episode, returning:
          - chains: per-player post-action state chains (for value network TD)
          - rewards: terminal rewards
          - opp_data: list of (encoded_state, action_index) for opponent model

        For opponent model training, we record EVERY action taken by BOTH players
        (since both players take turns being 'the opponent' to the other).
        We encode the state from the acting player's perspective, since the
        opponent model predicts what any player would do given their view.
        """
        self.game.reset()
        chains = [[], []]
        opp_data = []

        while not self.game.is_finished:
            current_player = self.game.current_player
            obs = self.game.get_observation(viewer_id=current_player)

            # Select action using training policy (1-ply with Boltzmann for training)
            action = self.agent.select_action_1ply(obs) if hasattr(self.agent, 'select_action_1ply') else self.agent.select_action(obs)
            if isinstance(action, tuple):
                action = action[0]

            # Record post-action state for the acting player (value network)
            post_obs, _ = LeducGame.simulate_action(obs, action)
            encoded = self.agent.encode_observation(post_obs, viewer_id=current_player)
            chains[current_player].append(encoded)

            # Record (state, action) for opponent model training
            # The opponent model predicts actions given a state encoding
            # from the acting player's perspective
            encoded_for_opp = self.agent.encode_observation(obs, viewer_id=current_player)
            opp_data.append((encoded_for_opp, action.value))

            self.game.step(action)

        rewards = self.game.get_reward()
        return chains, rewards, opp_data

    def update_model(self, batch_data: list) -> float:
        """
        Update both networks on a batch of episodes.

        Value network: TD(0) on per-player post-action chains.
        Opponent model: Cross-entropy on (state, action) pairs.
        """
        # --- Value network update (same as SelfPlayTrainer) ---
        self.value_optimizer.zero_grad()
        value_losses = []

        for episode in batch_data:
            chains, rewards = episode[0], episode[1]
            for p_idx in [0, 1]:
                chain = chains[p_idx]
                if not chain:
                    continue

                for t in range(len(chain)):
                    prediction = self.agent.model(chain[t]).squeeze(0)

                    if t == len(chain) - 1:
                        target = torch.FloatTensor([rewards[p_idx]])
                    else:
                        with torch.no_grad():
                            target = self.agent.model(chain[t + 1]).squeeze(0)

                    loss = self.value_criterion(prediction, target)
                    value_losses.append(loss)

        value_loss_val = 0.0
        if value_losses:
            mean_value_loss = torch.stack(value_losses).mean()
            mean_value_loss.backward()
            self.value_optimizer.step()
            value_loss_val = mean_value_loss.item()

        # --- Opponent model update ---
        self.opp_optimizer.zero_grad()
        opp_losses = []

        all_opp_states = []
        all_opp_actions = []
        for episode in batch_data:
            opp_data = episode[2]
            for encoded_state, action_idx in opp_data:
                all_opp_states.append(encoded_state.squeeze(0))
                all_opp_actions.append(action_idx)

        opp_loss_val = 0.0
        if all_opp_states:
            states_batch = torch.stack(all_opp_states)  # [N, 15]
            actions_batch = torch.LongTensor(all_opp_actions)  # [N]
            logits = self.agent.opponent_model(states_batch)  # [N, 3]
            opp_loss = self.opp_criterion(logits, actions_batch)
            opp_loss.backward()
            self.opp_optimizer.step()
            opp_loss_val = opp_loss.item()

        # Return combined loss for logging
        return value_loss_val + opp_loss_val

    def debug_episode(self) -> Dict:
        """
        Run one episode with debug trace showing 2-ply evaluations.
        """
        self.game.reset()
        episode_trace = []

        old_train_mode = self.agent.train_mode
        self.agent.set_train_mode(False)

        while not self.game.is_finished:
            current_player = self.game.current_player
            obs = self.game.get_observation(viewer_id=current_player)

            # Get 2-ply evaluations
            evaluations_2ply = self.agent.get_action_evaluations_2ply(obs)
            # Get 1-ply evaluations for comparison
            evaluations_1ply = self.agent.get_action_evaluations(obs)

            # Select action via 2-ply (greedy)
            selected_eval = max(evaluations_2ply, key=lambda x: x["value"])
            action = selected_eval["action"]

            step_info = {
                "player_id": current_player,
                "observation": {
                    "player_hand": obs.player_hand,
                    "board": obs.board,
                    "pot": obs.pot,
                    "current_round": obs.current_round,
                },
                "evaluations_2ply": [
                    {
                        "action": e["action"].name,
                        "value": e["value"],
                        "method": e.get("method", "unknown"),
                        "opp_probs": e.get("opp_probs", {}),
                    } for e in evaluations_2ply
                ],
                "evaluations_1ply": [
                    {
                        "action": e["action"].name,
                        "value": e["value"],
                    } for e in evaluations_1ply
                ],
                "selected_action": action.name,
            }
            episode_trace.append(step_info)
            self.game.step(action)

        rewards = self.game.get_reward()

        self.agent.set_train_mode(old_train_mode)
        return {
            "trace": episode_trace,
            "final_rewards": rewards,
            "eval_type": "opponent_model_2ply",
        }

    def update_params(self, params: Dict):
        """Updates learning rates for both optimizers."""
        if "lr" in params:
            new_lr = params["lr"]
            for param_group in self.value_optimizer.param_groups:
                param_group['lr'] = new_lr
            for param_group in self.opp_optimizer.param_groups:
                param_group['lr'] = new_lr
            print(f"Learning rate updated to: {new_lr}")
