"""
StateOnlyAgent — paper-canonical Ablation A
=============================================
Architecture:
    V(s) = V_base(s)  [frozen]  +  head(s)

  V_base : pretrained 15-dim value network (agents/value_based/checkpoint.pt), frozen
  head   : trainable 15→32→32→1 MLP (StateOnlyHead), no stats input

Ablation A: removes opponent statistics from the modulation head. Keeps all
else identical to FullModulationAgent. Isolates contribution of opponent
modeling — if this agent performs similarly to FullModulationAgent, the
opponent statistics provide no useful signal.
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))
sys.path.insert(0, ROOT)

from agents.value_based.agent import ValueNetwork
from agents.base import BaseAgent
from engine.leduc_game import Action, LeducGame
from engine.observation import Observation

GAME_DIMS = 15
STAT_DIMS = 7
DEFAULT_BASE_CKPT = os.path.join(ROOT, "agents", "value_based", "checkpoint.pt")

_CARD_MAP  = {'J': 0, 'Q': 1, 'K': 2}
_MAX_CHIPS = 13


class StateOnlyHead(nn.Module):
    """15-dim residual MLP: (15,) → 1 scalar. Near-zero output init."""
    def __init__(self, input_size: int = GAME_DIMS, hidden_size: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )
        # Near-zero output-layer init: training starts close to the frozen base.
        nn.init.uniform_(self.net[-1].weight, -0.01, 0.01)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class StateOnlyAgent(BaseAgent):
    """
    Frozen value base + trainable state-only residual head.
    V(s) = V_base(s) + head(s)

    opp_stats is accepted but IGNORED in select_action and compute_value.
    """

    def __init__(
        self,
        base_ckpt: str = None,
        head_ckpt: str = None,
        temperature: float = 1.0,
    ):
        self.temperature = temperature
        self.train_mode  = False

        # ── frozen base ───────────────────────────────────────────────────────
        self.base = ValueNetwork(GAME_DIMS, hidden_size=64)
        ckpt_path = base_ckpt or DEFAULT_BASE_CKPT
        self.base.load_state_dict(
            torch.load(ckpt_path, map_location="cpu", weights_only=True)
        )
        for p in self.base.parameters():
            p.requires_grad = False
        self.base.eval()

        # ── trainable state-only head ─────────────────────────────────────────
        self.head = StateOnlyHead(GAME_DIMS, hidden_size=32)
        if head_ckpt:
            self.head.load_state_dict(
                torch.load(head_ckpt, map_location="cpu", weights_only=True)
            )
        self.head.eval()

    # ── mode control ──────────────────────────────────────────────────────────

    def set_train_mode(self, mode: bool):
        self.train_mode = mode
        self.head.train(mode)   # base stays frozen regardless

    # ── persistence ───────────────────────────────────────────────────────────

    def save_model(self, path: str):
        torch.save(self.head.state_dict(), path)

    def load_model(self, path: str):
        self.head.load_state_dict(
            torch.load(path, map_location="cpu", weights_only=True)
        )

    # ── encoding ──────────────────────────────────────────────────────────────

    def _encode_game(self, obs: Observation, viewer_id: int) -> torch.Tensor:
        """15-dim game-state encoding."""
        hand_idx = _CARD_MAP.get(obs.player_hand)
        hand_vec = torch.zeros(3)
        if hand_idx is not None:
            hand_vec[hand_idx] = 1.0
        board_idx = _CARD_MAP.get(obs.board, 3)
        board_vec = torch.zeros(4)
        board_vec[board_idx] = 1.0
        p0, p1 = obs.pot
        pot_rel = [p0, p1] if viewer_id == 0 else [p1, p0]
        pot_vec = torch.tensor(pot_rel, dtype=torch.float32) / _MAX_CHIPS
        feats = torch.tensor([
            1.0 if viewer_id == obs.current_player else 0.0,
            float(viewer_id),
            float(obs.current_round),
            1.0 if obs.is_finished else 0.0,
            1.0 if (obs.board is not None and obs.player_hand == obs.board) else 0.0,
            obs.raises_this_round / 2.0,
        ])
        return torch.cat([hand_vec, board_vec, pot_vec, feats])   # (15,)

    # ── value computation ─────────────────────────────────────────────────────

    def compute_value(
        self,
        game_enc: torch.Tensor,    # (1, 15)
        opp_stats=None,            # ignored
    ) -> torch.Tensor:
        """V(s) = V_base(s) + head(s).  (1, 1) output. opp_stats ignored."""
        with torch.no_grad():
            v_base = self.base(game_enc)
        delta = self.head(game_enc)                     # (1, 1)
        return v_base + delta

    # ── action selection ──────────────────────────────────────────────────────

    def _get_action_values(self, obs: Observation) -> list:
        cp = obs.current_player
        results = []
        for action in obs.legal_actions:
            post_obs, done = LeducGame.simulate_action(obs, action)
            if done and action == Action.FOLD:
                val = -float(obs.pot[cp])
            else:
                game_enc = self._encode_game(post_obs, cp).unsqueeze(0)
                with torch.no_grad():
                    val = self.compute_value(game_enc).item()
            results.append({"action": action, "value": val,
                            "game_enc": self._encode_game(post_obs, cp)})
        return results

    def select_action(
        self,
        obs: Observation,
        opp_stats=None,   # accepted but ignored
    ) -> Action:
        results = self._get_action_values(obs)
        if not results:
            return Action.FOLD
        if self.train_mode:
            values = torch.tensor([r["value"] for r in results])
            probs  = torch.softmax(values / self.temperature, dim=0)
            idx    = torch.multinomial(probs, 1).item()
            return results[idx]["action"]
        return max(results, key=lambda r: r["value"])["action"]
