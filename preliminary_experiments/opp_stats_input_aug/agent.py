"""
StatAugValueAgent
==================
ValueBasedAgent extended with 7 opponent-statistics features.

Input:  22 dims = 15 (game state, identical encoding to baseline) + 7 (opponent stats)
Model:  ValueNetwork(22, hidden_size=64)  — same width as baseline
Output: scalar V(s, opp_stats)

The 7 opponent stats are provided externally (from OpponentStatsTracker.get_features())
and appended to the game-state encoding at inference / training time.
"""

import numpy as np
import torch

from agents.base import BaseAgent
from agents.value_based.agent import ValueNetwork   # reuse the same MLP class
from engine.leduc_game import Action, LeducGame
from engine.observation import Observation

from .stats_tracker import N_STATS


class StatAugValueAgent(BaseAgent):
    CARD_MAP  = {'J': 0, 'Q': 1, 'K': 2}
    MAX_CHIPS = 13
    GAME_DIMS = 15
    STAT_DIMS = N_STATS        # 7
    INPUT_DIMS = GAME_DIMS + STAT_DIMS  # 22

    def __init__(self, model_path: str = None, temperature: float = 1.0):
        self.input_size  = self.INPUT_DIMS
        self.temperature = temperature
        self.train_mode  = False
        self.model = ValueNetwork(self.INPUT_DIMS, hidden_size=64)
        if model_path:
            self.load_model(model_path)
        self.model.eval()

    # ── persistence ───────────────────────────────────────────────────────────

    def set_train_mode(self, mode: bool):
        self.train_mode = mode
        self.model.train(mode)

    def save_model(self, path: str):
        torch.save(self.model.state_dict(), path)

    def load_model(self, path: str):
        self.model.load_state_dict(
            torch.load(path, map_location="cpu", weights_only=True)
        )

    # ── encoding ──────────────────────────────────────────────────────────────

    def _encode_game(self, obs: Observation, viewer_id: int) -> torch.Tensor:
        """Identical 15-dim game encoding as ValueBasedAgent."""
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

        feats = torch.tensor([
            1.0 if viewer_id == obs.current_player else 0.0,
            float(viewer_id),
            float(obs.current_round),
            1.0 if obs.is_finished else 0.0,
            1.0 if (obs.board is not None and obs.player_hand == obs.board) else 0.0,
            obs.raises_this_round / 2.0,
        ])
        return torch.cat([hand_vec, board_vec, pot_vec, feats])   # shape: (15,)

    def encode_observation(self, obs: Observation, viewer_id: int,
                           opp_stats: np.ndarray) -> torch.Tensor:
        """Return 1×22 tensor: [15 game dims | 7 opponent-stat dims]."""
        game = self._encode_game(obs, viewer_id)
        stats = torch.tensor(opp_stats, dtype=torch.float32)
        return torch.cat([game, stats]).unsqueeze(0)   # shape: (1, 22)

    # ── inference ─────────────────────────────────────────────────────────────

    def _get_value(self, obs: Observation, viewer_id: int,
                   opp_stats: np.ndarray) -> float:
        enc = self.encode_observation(obs, viewer_id, opp_stats)
        with torch.no_grad():
            return self.model(enc).item()

    def get_action_evaluations(self, obs: Observation,
                               opp_stats: np.ndarray) -> list:
        cp = obs.current_player
        results = []
        for action in obs.legal_actions:
            post_obs, done = LeducGame.simulate_action(obs, action)
            if done and action == Action.FOLD:
                val = -float(obs.pot[cp])
            else:
                val = self._get_value(post_obs, viewer_id=cp, opp_stats=opp_stats)
            encoded = self.encode_observation(post_obs, viewer_id=cp,
                                              opp_stats=opp_stats)
            results.append({"action": action, "value": val, "encoded": encoded})
        return results

    def select_action(self, obs: Observation,
                      opp_stats: np.ndarray = None) -> Action:
        if opp_stats is None:
            opp_stats = np.full(self.STAT_DIMS, 0.5, dtype=np.float32)
        results = self.get_action_evaluations(obs, opp_stats)
        if not results:
            return Action.FOLD
        if self.train_mode:
            values = torch.tensor([r["value"] for r in results])
            probs  = torch.softmax(values / self.temperature, dim=0)
            idx    = torch.multinomial(probs, 1).item()
            return results[idx]["action"]
        return max(results, key=lambda x: x["value"])["action"]
