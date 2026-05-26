"""Agents for repr_policy_v1: policy learning on contrastive representation.

Three variants:
  - VanillaPolicyAgent: baseline REINFORCE on raw 15-dim features
  - ReprPolicyAgent: frozen contrastive encoder + REINFORCE policy head
  - ReprPolicyFineTuneAgent: unfrozen contrastive encoder + REINFORCE policy head
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple

from agents.base import BaseAgent
from engine.leduc_game import Action
from engine.observation import Observation

# Import ContrastiveEncoder from contrastive_repr_v1
from experiments.representation_learning.contrastive_repr_v1.agent import ContrastiveEncoder


class VanillaPolicyAgent(BaseAgent):
    """Baseline REINFORCE agent operating on raw 15-dim observation encoding.

    Policy network: 15 → 64 → 64 → 3 (same architecture as legacy policy_gradient.py)
    """

    CARD_MAP = {'J': 0, 'Q': 1, 'K': 2}
    MAX_CHIPS = 13

    def __init__(self, model_path: str = None):
        self.train_mode = False
        self.policy = nn.Sequential(
            nn.Linear(15, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 3),
        )
        if model_path:
            self.load_model(model_path)

    def encode_observation(self, obs: Observation, viewer_id: int = None) -> torch.Tensor:
        """Encode observation to 15-dim vector. Matches contrastive_repr_v1 encoding."""
        if viewer_id is None:
            viewer_id = obs.current_player

        hand_idx = self.CARD_MAP.get(obs.player_hand)
        hand_vec = torch.zeros(3)
        if hand_idx is not None:
            hand_vec[hand_idx] = 1.0

        board_idx = self.CARD_MAP.get(obs.board, 3)
        board_vec = torch.zeros(4)
        board_vec[board_idx] = 1.0

        p0_pot, p1_pot = obs.pot
        pot_rel = [p0_pot, p1_pot] if viewer_id == 0 else [p1_pot, p0_pot]
        pot_vec = torch.tensor(pot_rel, dtype=torch.float32) / self.MAX_CHIPS

        features = torch.tensor([
            1.0 if viewer_id == obs.current_player else 0.0,
            float(viewer_id),
            float(obs.current_round),
            1.0 if obs.is_finished else 0.0,
            1.0 if (obs.board is not None and obs.player_hand == obs.board) else 0.0,
            obs.raises_this_round / 2.0,
        ])

        return torch.cat([hand_vec, board_vec, pot_vec, features]).unsqueeze(0)

    def _get_legal_logits(self, obs: Observation, viewer_id: int = None) -> torch.Tensor:
        """Forward pass through policy; mask illegal actions with -inf."""
        encoded = self.encode_observation(obs, viewer_id=viewer_id)
        logits = self.policy(encoded).squeeze(0)  # [3]

        legal_mask = torch.full((3,), float('-inf'))
        for action in obs.legal_actions:
            legal_mask[action.value] = 0.0
        return logits + legal_mask

    def select_action_with_log_prob(
        self, obs: Observation, viewer_id: int = None
    ) -> Tuple[Action, torch.Tensor]:
        """Sample action and return (action, log_prob). Used during training."""
        masked_logits = self._get_legal_logits(obs, viewer_id=viewer_id)
        dist = torch.distributions.Categorical(logits=masked_logits)
        action_idx = dist.sample()
        return Action(action_idx.item()), dist.log_prob(action_idx)

    def select_action(self, obs: Observation) -> Action:
        """Greedy action selection for evaluation."""
        with torch.no_grad():
            masked_logits = self._get_legal_logits(obs)
            if self.train_mode:
                dist = torch.distributions.Categorical(logits=masked_logits)
                return Action(dist.sample().item())
            else:
                return Action(masked_logits.argmax().item())

    def set_train_mode(self, mode: bool):
        self.train_mode = mode
        self.policy.train(mode)

    def save_model(self, path: str) -> None:
        torch.save({'policy': self.policy.state_dict()}, path)

    def load_model(self, path: str) -> None:
        state = torch.load(path, weights_only=False)
        if 'policy' in state:
            self.policy.load_state_dict(state['policy'])
        else:
            # Fallback: assume raw state dict
            self.policy.load_state_dict(state)

    def parameters(self):
        return self.policy.parameters()


class ReprPolicyAgent(BaseAgent):
    """REINFORCE agent with a FROZEN contrastive encoder.

    Architecture: raw 15-dim → ContrastiveEncoder (frozen) → 8-dim → policy head (8→64→64→3)
    """

    CARD_MAP = {'J': 0, 'Q': 1, 'K': 2}
    MAX_CHIPS = 13

    def __init__(self, checkpoint_path: str, model_path: str = None,
                 finetune_encoder: bool = False):
        self.train_mode = False
        self.finetune_encoder = finetune_encoder

        # Load encoder from checkpoint
        self.encoder = ContrastiveEncoder(input_size=15, hidden_size=64, embedding_dim=8)
        ckpt = torch.load(checkpoint_path, weights_only=False)
        self.encoder.load_state_dict(ckpt['encoder'])

        # Freeze encoder unless finetuning
        if not finetune_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

        # Policy head: 8 → 64 → 64 → 3
        self.policy_head = nn.Sequential(
            nn.Linear(8, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 3),
        )

        if model_path:
            self.load_model(model_path)

    def encode_observation(self, obs: Observation, viewer_id: int = None) -> torch.Tensor:
        """Encode observation to 15-dim vector (exact copy from contrastive_repr_v1)."""
        if viewer_id is None:
            viewer_id = obs.current_player

        hand_idx = self.CARD_MAP.get(obs.player_hand)
        hand_vec = torch.zeros(3)
        if hand_idx is not None:
            hand_vec[hand_idx] = 1.0

        board_idx = self.CARD_MAP.get(obs.board, 3)
        board_vec = torch.zeros(4)
        board_vec[board_idx] = 1.0

        p0_pot, p1_pot = obs.pot
        pot_rel = [p0_pot, p1_pot] if viewer_id == 0 else [p1_pot, p0_pot]
        pot_vec = torch.tensor(pot_rel, dtype=torch.float32) / self.MAX_CHIPS

        features = torch.tensor([
            1.0 if viewer_id == obs.current_player else 0.0,
            float(viewer_id),
            float(obs.current_round),
            1.0 if obs.is_finished else 0.0,
            1.0 if (obs.board is not None and obs.player_hand == obs.board) else 0.0,
            obs.raises_this_round / 2.0,
        ])

        return torch.cat([hand_vec, board_vec, pot_vec, features]).unsqueeze(0)

    def _get_legal_logits(self, obs: Observation, viewer_id: int = None) -> torch.Tensor:
        """Encode → embed → policy head → mask illegal actions."""
        raw = self.encode_observation(obs, viewer_id=viewer_id)
        z = self.encoder(raw)  # [1, 8]
        logits = self.policy_head(z).squeeze(0)  # [3]

        legal_mask = torch.full((3,), float('-inf'))
        for action in obs.legal_actions:
            legal_mask[action.value] = 0.0
        return logits + legal_mask

    def select_action_with_log_prob(
        self, obs: Observation, viewer_id: int = None
    ) -> Tuple[Action, torch.Tensor]:
        """Sample action and return (action, log_prob). Used during training."""
        masked_logits = self._get_legal_logits(obs, viewer_id=viewer_id)
        dist = torch.distributions.Categorical(logits=masked_logits)
        action_idx = dist.sample()
        return Action(action_idx.item()), dist.log_prob(action_idx)

    def select_action(self, obs: Observation) -> Action:
        """Greedy action selection for evaluation."""
        with torch.no_grad():
            masked_logits = self._get_legal_logits(obs)
            if self.train_mode:
                dist = torch.distributions.Categorical(logits=masked_logits)
                return Action(dist.sample().item())
            else:
                return Action(masked_logits.argmax().item())

    def set_train_mode(self, mode: bool):
        self.train_mode = mode
        self.encoder.train(mode if self.finetune_encoder else False)
        self.policy_head.train(mode)

    def save_model(self, path: str) -> None:
        state = {
            'policy_head': self.policy_head.state_dict(),
            'finetune_encoder': self.finetune_encoder,
        }
        if self.finetune_encoder:
            state['encoder'] = self.encoder.state_dict()
        torch.save(state, path)

    def load_model(self, path: str) -> None:
        state = torch.load(path, weights_only=False)
        self.policy_head.load_state_dict(state['policy_head'])
        if self.finetune_encoder and 'encoder' in state:
            self.encoder.load_state_dict(state['encoder'])

    def parameters(self):
        if self.finetune_encoder:
            return list(self.encoder.parameters()) + list(self.policy_head.parameters())
        return self.policy_head.parameters()


class ReprPolicyFineTuneAgent(ReprPolicyAgent):
    """REINFORCE agent with an UNFROZEN (finetunable) contrastive encoder.

    Same as ReprPolicyAgent but encoder.requires_grad=True.
    """

    def __init__(self, checkpoint_path: str, model_path: str = None):
        super().__init__(
            checkpoint_path=checkpoint_path,
            model_path=model_path,
            finetune_encoder=True,
        )
