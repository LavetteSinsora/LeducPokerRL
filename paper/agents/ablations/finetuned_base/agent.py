"""
FinetunedBaseAgent — paper-canonical Ablation B
=================================================
Architecture:
    V(s, opp) = V_base(s)  [UNFROZEN]  +  Δ(s, opp_stats)

  V_base : pretrained 15-dim value network (agents/value_based/checkpoint.pt),
           UNFROZEN — parameters update during training
  Δ      : trainable 22→32→32→1 ModulationHead (same as FullModulationAgent)

Ablation B: base network is unfrozen and updates during training alongside
the modulation head. Isolates the effect of freezing the pre-trained base.
If this agent performs worse than FullModulationAgent, it confirms that
freezing the base provides a stable inductive bias that benefits training.

Exposes base_drift() to measure how much the base has drifted from its
initialization during training.
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
MOD_INPUT_DIMS = GAME_DIMS + STAT_DIMS   # 22
DEFAULT_BASE_CKPT = os.path.join(ROOT, "agents", "value_based", "checkpoint.pt")

_CARD_MAP  = {'J': 0, 'Q': 1, 'K': 2}
_MAX_CHIPS = 13


class ModulationHead(nn.Module):
    """Lightweight residual MLP: (22,) → 1 scalar. Near-zero output init."""
    def __init__(self, input_size: int = MOD_INPUT_DIMS, hidden_size: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )
        nn.init.uniform_(self.net[-1].weight, -0.01, 0.01)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class FinetunedBaseAgent(BaseAgent):
    """
    Unfrozen value base + trainable ungated residual.
    V(s, opp) = V_base(s) + Δ(s, opp_stats)

    Gradients flow through both base and mod during training.
    """

    def __init__(
        self,
        base_ckpt: str = None,
        mod_ckpt: str = None,
        temperature: float = 1.0,
    ):
        self.temperature = temperature
        self.train_mode  = False

        # ── UNFROZEN base ─────────────────────────────────────────────────────
        self.base = ValueNetwork(GAME_DIMS, hidden_size=64)
        ckpt_path = base_ckpt or DEFAULT_BASE_CKPT
        self.base.load_state_dict(
            torch.load(ckpt_path, map_location="cpu", weights_only=True)
        )
        # NOTE: requires_grad = True (default) — base is NOT frozen
        self.base.train(False)

        # Snapshot initial base weights for drift measurement
        self._base_init_params = [p.data.clone() for p in self.base.parameters()]

        # ── trainable residual head ───────────────────────────────────────────
        self.mod = ModulationHead(MOD_INPUT_DIMS, hidden_size=32)
        if mod_ckpt:
            self.mod.load_state_dict(
                torch.load(mod_ckpt, map_location="cpu", weights_only=True)
            )
        self.mod.eval()

    # ── mode control ──────────────────────────────────────────────────────────

    def set_train_mode(self, mode: bool):
        self.train_mode = mode
        self.base.train(mode)
        self.mod.train(mode)

    # ── persistence ───────────────────────────────────────────────────────────

    def save_model(self, path: str):
        """Save both base and mod state_dicts."""
        torch.save({
            "base": self.base.state_dict(),
            "mod":  self.mod.state_dict(),
        }, path)

    def load_model(self, path: str):
        state = torch.load(path, map_location="cpu", weights_only=True)
        self.base.load_state_dict(state["base"])
        self.mod.load_state_dict(state["mod"])
        # Update init snapshot after loading
        self._base_init_params = [p.data.clone() for p in self.base.parameters()]

    # ── drift measurement ─────────────────────────────────────────────────────

    def base_drift(self) -> float:
        """Mean L2 distance of current base weights from initialization snapshot."""
        total = 0.0
        count = 0
        for p, p0 in zip(self.base.parameters(), self._base_init_params):
            total += (p.data - p0).norm().item()
            count += 1
        return total / count if count > 0 else 0.0

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
        game_enc: torch.Tensor,   # (1, 15)
        stats_vec: torch.Tensor,  # (1, 7)
    ) -> torch.Tensor:
        """V(s, opp) = V_base(s) + Δ(s, opp_stats).  (1, 1) output.
        Gradients flow through both base and mod."""
        v_base  = self.base(game_enc)                               # (1, 1), grad flows
        mod_inp = torch.cat([game_enc, stats_vec], dim=1)           # (1, 22)
        delta   = self.mod(mod_inp)                                 # (1, 1)
        return v_base + delta

    # ── action selection ──────────────────────────────────────────────────────

    def _get_action_values(
        self,
        obs: Observation,
        opp_stats: np.ndarray,
    ) -> list:
        cp = obs.current_player
        results = []
        for action in obs.legal_actions:
            post_obs, done = LeducGame.simulate_action(obs, action)
            if done and action == Action.FOLD:
                val = -float(obs.pot[cp])
            else:
                game_enc  = self._encode_game(post_obs, cp).unsqueeze(0)
                stats_t   = torch.tensor(opp_stats, dtype=torch.float32).unsqueeze(0)
                with torch.no_grad():
                    val = self.compute_value(game_enc, stats_t).item()
            results.append({"action": action, "value": val,
                            "game_enc": self._encode_game(post_obs, cp)})
        return results

    def select_action(
        self,
        obs: Observation,
        opp_stats: np.ndarray = None,
    ) -> Action:
        if opp_stats is None:
            opp_stats = np.full(STAT_DIMS, 0.5, dtype=np.float32)
        results = self._get_action_values(obs, opp_stats)
        if not results:
            return Action.FOLD
        if self.train_mode:
            values = torch.tensor([r["value"] for r in results])
            probs  = torch.softmax(values / self.temperature, dim=0)
            idx    = torch.multinomial(probs, 1).item()
            return results[idx]["action"]
        return max(results, key=lambda r: r["value"])["action"]
