"""
GatedModulationAgent — DALI_modulation
========================================
Architecture:
    V(s, opp) = V_base(s)  [frozen]
              + g(s, opp_stats) × Δ(s, opp_stats)

  V_base   : pretrained 15-dim value network (agents/value_based/checkpoint.pt), frozen
  g        : GateNet (22→16→16→1→sigmoid) — state-conditioned gate ∈ [0, 1]
  Δ        : ModNet  (22→32→32→1) — opponent-specific residual

Key difference from canonical `modulated_value` (agents/modulated_value/):
  - Gate input: 22-dim (game_enc + stats) vs 4-dim (stats only) in canonical agent
  - This allows the gate to learn which game states benefit from modulation,
    directly leveraging the EV_variation_analysis finding that 51.65% of states
    have opponent-driven action switching while others are opponent-invariant.

During cold-start (confidence ≈ 0), pool-mean stats are near 0.5, so the gate
sees a low-information stats vector. A well-trained gate can learn to suppress
modulation in those early hands, falling back to the frozen base.
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, ROOT)

from agents.value_based.agent import ValueNetwork
from agents.base import BaseAgent
from engine.leduc_game import Action, LeducGame
from engine.observation import Observation

GAME_DIMS      = 15
STAT_DIMS      = 7
MOD_INPUT_DIMS = GAME_DIMS + STAT_DIMS   # 22
DEFAULT_BASE_CKPT = os.path.join(ROOT, "agents", "value_based", "checkpoint.pt")

_CARD_MAP  = {'J': 0, 'Q': 1, 'K': 2}
_MAX_CHIPS = 13


class GateNet(nn.Module):
    """
    State-conditioned gate: (22,) → scalar ∈ [0, 1].
    Input: concatenation of 15-dim game encoding and 7-dim opponent stats.
    Near-zero output-layer init so the gate starts near sigmoid(0) ≈ 0.5 and
    can learn to push toward 0 (suppress) or 1 (amplify) based on state type.
    """
    def __init__(self, input_size: int = MOD_INPUT_DIMS, hidden_size: int = 16):
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
        return torch.sigmoid(self.net(x))   # (batch, 1), ∈ [0, 1]


class ModNet(nn.Module):
    """Residual MLP: (22,) → 1 scalar."""
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


class GatedModulationAgent(BaseAgent):
    """
    Frozen value base + state-conditioned gate × residual.
    V(s, opp) = V_base(s) + g(s, stats) * Δ(s, stats)
    """

    def __init__(
        self,
        base_ckpt: str = None,
        gate_ckpt: str = None,
        mod_ckpt: str = None,
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

        # ── trainable gate and modulation nets ────────────────────────────────
        self.gate_net = GateNet(MOD_INPUT_DIMS, hidden_size=16)
        self.mod_net  = ModNet(MOD_INPUT_DIMS, hidden_size=32)

        if gate_ckpt:
            self.gate_net.load_state_dict(
                torch.load(gate_ckpt, map_location="cpu", weights_only=True))
        if mod_ckpt:
            self.mod_net.load_state_dict(
                torch.load(mod_ckpt, map_location="cpu", weights_only=True))
        self.gate_net.eval()
        self.mod_net.eval()

    # ── mode control ──────────────────────────────────────────────────────────

    def set_train_mode(self, mode: bool):
        self.train_mode = mode
        self.gate_net.train(mode)
        self.mod_net.train(mode)

    # ── persistence ───────────────────────────────────────────────────────────

    def save_model(self, path: str):
        """Save both gate and mod networks as a combined state dict."""
        torch.save({
            "gate_net": self.gate_net.state_dict(),
            "mod_net":  self.mod_net.state_dict(),
        }, path)

    def load_model(self, path: str):
        state = torch.load(path, map_location="cpu", weights_only=True)
        self.gate_net.load_state_dict(state["gate_net"])
        self.mod_net.load_state_dict(state["mod_net"])

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
        stats_vec: torch.Tensor,   # (1, 7)
    ) -> torch.Tensor:
        """
        V(s, opp) = V_base(s) + g(s, stats) * Δ(s, stats).
        Gradients flow through gate_net and mod_net only (base is frozen).
        Returns (1, 1) tensor.
        """
        with torch.no_grad():
            v_base = self.base(game_enc)                            # (1, 1), no grad
        mod_inp = torch.cat([game_enc, stats_vec], dim=1)           # (1, 22)
        gate    = self.gate_net(mod_inp)                            # (1, 1), ∈ [0,1]
        delta   = self.mod_net(mod_inp)                             # (1, 1)
        return v_base + gate * delta

    def get_gate_value(
        self,
        game_enc: torch.Tensor,
        stats_vec: torch.Tensor,
    ) -> float:
        """Inspect gate activation for a given (state, stats) pair."""
        mod_inp = torch.cat([game_enc, stats_vec], dim=1)
        with torch.no_grad():
            return self.gate_net(mod_inp).item()

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
