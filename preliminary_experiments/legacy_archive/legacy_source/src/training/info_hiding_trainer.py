"""
Trainer for the Information-Hiding Agent.

Alternates between:
  1. Training the spy network to predict the agent's hand from its action
     sequence (standard cross-entropy).
  2. Training the actor-critic with an adversarial spy loss that pushes the
     policy toward actions the spy cannot read.

Loss for the actor-critic:
    L = policy_gradient_loss + alpha * value_loss - lambda * spy_cross_entropy

The NEGATIVE sign on the spy term means the policy is rewarded when the spy
fails to predict the hand.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict
from src.engine.leduc_game import LeducGame, Action
from src.agents.base import BaseAgent
from src.training.base import BaseTrainer


# Action-sequence encoding helper
ACTION_MAP = {Action.FOLD: 0, Action.CALL: 1, Action.RAISE: 2}
CARD_IDX = {'J': 0, 'Q': 1, 'K': 2}
MAX_ACTION_SLOTS = 4   # max actions one player can take in a Leduc hand
FEATURES_PER_SLOT = 4  # fold / call / raise / no-action-yet
SPY_INPUT_DIM = MAX_ACTION_SLOTS * FEATURES_PER_SLOT + 4  # 20


def encode_action_sequence(actions: list, pot: list, round_reached: int,
                           total_raises: int) -> torch.Tensor:
    """Encode a player's action sequence into the spy's 20-dim input.

    Args:
        actions: list of Action enums taken by the agent (up to 4).
        pot: [my_pot, opp_pot] at end of hand (raw chip counts).
        round_reached: 0 (preflop only) or 1 (reached flop).
        total_raises: total raises across the hand.

    Returns:
        Tensor of shape (20,).
    """
    vec = torch.zeros(SPY_INPUT_DIM)

    for i, action in enumerate(actions[:MAX_ACTION_SLOTS]):
        base = i * FEATURES_PER_SLOT
        vec[base + ACTION_MAP[action]] = 1.0

    # Slots without actions get a "no-action-yet" flag
    for i in range(len(actions), MAX_ACTION_SLOTS):
        base = i * FEATURES_PER_SLOT
        vec[base + 3] = 1.0  # no-action-yet indicator

    offset = MAX_ACTION_SLOTS * FEATURES_PER_SLOT  # 16
    max_chips = 13.0
    vec[offset] = pot[0] / max_chips
    vec[offset + 1] = pot[1] / max_chips
    vec[offset + 2] = float(round_reached)
    vec[offset + 3] = total_raises / 4.0  # max 4 raises per hand (2 per round)

    return vec


class InfoHidingTrainer(BaseTrainer):
    """
    Trains an InfoHidingAgent using REINFORCE + value baseline + adversarial
    spy loss.

    Hyperparameters:
        learning_rate:  shared lr for actor-critic optimiser (default 1e-4).
        spy_lr:         learning rate for the spy network (default 1e-3).
        value_coeff:    weight on value loss (alpha, default 0.5).
        info_hiding_coeff: weight on spy loss (lambda, default 0.1).
    """

    def __init__(self, agent: BaseAgent, learning_rate: float = 1e-4,
                 spy_lr: float = 1e-3, value_coeff: float = 0.5,
                 info_hiding_coeff: float = 0.1):
        super().__init__(agent, eval_interval=500, eval_num_games=200)
        self.ac_optimizer = optim.Adam(agent.model.parameters(), lr=learning_rate)
        self.spy_optimizer = optim.Adam(agent.spy.parameters(), lr=spy_lr)
        self.value_coeff = value_coeff
        self.info_hiding_coeff = info_hiding_coeff
        self.game = LeducGame()
        self.spy_loss_fn = nn.CrossEntropyLoss()

    # ── episode collection ──────────────────────────────────────────

    def collect_episode(self) -> dict:
        """Play one self-play game, recording everything needed for both the
        actor-critic update and the spy update.

        Returns dict with:
            log_probs[player]: list of log-prob tensors
            values[player]:    list of V(s) tensors
            rewards:           [reward_p0, reward_p1]
            hand_cards:        [card_p0, card_p1]  (str)
            action_seqs:       [actions_p0, actions_p1]  (list of Action)
            pot:               final pot [p0, p1]
            round_reached:     int  (0 or 1)
            total_raises:      int
        """
        self.game.reset()

        log_probs = [[], []]
        values = [[], []]
        action_seqs = [[], []]

        self.agent.set_train_mode(True)

        total_raises = 0

        while not self.game.is_finished:
            player = self.game.current_player
            obs = self.game.get_observation(viewer_id=player)
            encoded = self.agent.encode_observation(obs)

            probs, value = self.agent.model(encoded)
            probs = probs.squeeze(0)
            value = value.squeeze(0)

            # Legal masking
            legal_mask = torch.zeros(3)
            for action in obs.legal_actions:
                legal_mask[action.value] = 1.0
            probs = probs * legal_mask
            probs = probs / probs.sum()

            dist = torch.distributions.Categorical(probs)
            action_idx = dist.sample()
            action = Action(action_idx.item())

            log_probs[player].append(dist.log_prob(action_idx))
            values[player].append(value)
            action_seqs[player].append(action)

            if action == Action.RAISE:
                total_raises += 1

            self.game.step(action)

        rewards = self.game.get_reward()

        return {
            "log_probs": log_probs,
            "values": values,
            "rewards": rewards,
            "hand_cards": list(self.game.player_hands),
            "action_seqs": action_seqs,
            "pot": list(self.game.pot),
            "round_reached": self.game.current_round,
            "total_raises": total_raises,
        }

    # ── model update ────────────────────────────────────────────────

    def update_model(self, batch_data: list) -> float:
        """
        Two-phase update:
          Phase 1 — train spy on completed episodes.
          Phase 2 — train actor-critic with adversarial spy term.

        Returns the actor-critic total loss (float).
        """

        # ── Phase 1: Spy update ─────────────────────────────────────
        self.spy_optimizer.zero_grad()
        spy_loss_total = torch.tensor(0.0)
        spy_count = 0

        for episode in batch_data:
            for player in [0, 1]:
                if not episode["action_seqs"][player]:
                    continue
                spy_input = encode_action_sequence(
                    episode["action_seqs"][player],
                    episode["pot"],
                    episode["round_reached"],
                    episode["total_raises"],
                ).unsqueeze(0)

                hand_label = torch.tensor([CARD_IDX[episode["hand_cards"][player]]])
                spy_logits = self.agent.spy(spy_input)
                spy_loss_total = spy_loss_total + self.spy_loss_fn(spy_logits, hand_label)
                spy_count += 1

        if spy_count > 0:
            spy_loss_avg = spy_loss_total / spy_count
            spy_loss_avg.backward()
            self.spy_optimizer.step()

        # ── Phase 2: Actor-Critic update with adversarial spy term ──
        self.ac_optimizer.zero_grad()
        ac_loss_total = torch.tensor(0.0)

        for episode in batch_data:
            log_probs = episode["log_probs"]
            values = episode["values"]
            rewards = episode["rewards"]

            for player in [0, 1]:
                if not log_probs[player]:
                    continue

                reward = rewards[player]

                # Standard REINFORCE + value loss
                for lp, v in zip(log_probs[player], values[player]):
                    advantage = reward - v.detach()
                    policy_loss = -lp * advantage
                    value_loss = (v - reward) ** 2
                    ac_loss_total = ac_loss_total + policy_loss + self.value_coeff * value_loss

                # Adversarial spy term: compute spy cross-entropy and SUBTRACT it
                if episode["action_seqs"][player]:
                    spy_input = encode_action_sequence(
                        episode["action_seqs"][player],
                        episode["pot"],
                        episode["round_reached"],
                        episode["total_raises"],
                    ).unsqueeze(0)

                    hand_label = torch.tensor([CARD_IDX[episode["hand_cards"][player]]])
                    # Spy weights are frozen for this pass (we only want gradients
                    # to flow through the policy, not the spy).
                    with torch.no_grad():
                        spy_logits = self.agent.spy(spy_input)
                    spy_ce = self.spy_loss_fn(spy_logits, hand_label)

                    # Subtract: lower spy CE (=spy doing well) should be penalised
                    # by adding +lambda*spy_accuracy_proxy = -lambda*spy_CE
                    # Wait -- we want the policy to make the spy FAIL, i.e. spy CE
                    # should be HIGH. So we SUBTRACT spy_CE from the loss, meaning
                    # the optimiser drives spy CE higher (spy accuracy lower).
                    ac_loss_total = ac_loss_total - self.info_hiding_coeff * spy_ce

        ac_loss_total = ac_loss_total / len(batch_data)
        ac_loss_total.backward()
        self.ac_optimizer.step()

        return ac_loss_total.item()

    # ── parameter update from dashboard ─────────────────────────────

    def update_params(self, params: Dict):
        if "lr" in params:
            for pg in self.ac_optimizer.param_groups:
                pg["lr"] = params["lr"]
        if "value_coeff" in params:
            self.value_coeff = params["value_coeff"]
        if "info_hiding_coeff" in params:
            self.info_hiding_coeff = params["info_hiding_coeff"]

    # ── debug trace for analyzer ────────────────────────────────────

    def debug_episode(self) -> Dict:
        self.game.reset()
        episode_trace = []
        old_train_mode = self.agent.train_mode
        self.agent.set_train_mode(False)

        while not self.game.is_finished:
            current_player = self.game.current_player
            obs = self.game.get_observation(viewer_id=current_player)
            evaluations = self.agent.get_action_evaluations(obs)

            selected_eval = max(evaluations, key=lambda x: x["probability"])
            action = selected_eval["action"]

            step_info = {
                "player_id": current_player,
                "observation": {
                    "player_hand": obs.player_hand,
                    "board": obs.board,
                    "pot": obs.pot,
                    "current_round": obs.current_round,
                },
                "evaluations": [
                    {
                        "action": e["action"].name,
                        "action_id": e["action"].value,
                        "probability": e["probability"],
                        "raw_probability": e["raw_probability"],
                        "value_estimate": e["value_estimate"],
                    }
                    for e in evaluations
                ],
                "selected_action": action.name,
                "selected_action_id": action.value,
                "value_estimate": selected_eval["value_estimate"],
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
            "eval_type": "info_hiding",
        }
