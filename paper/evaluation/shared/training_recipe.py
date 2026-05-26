"""
Shared training utilities for opp_stats_modulation_v2.

Provides:
  WeightedOpponentSampler — weighted random sampling of opponents per session.
      CFR and heuristic are oversampled 3× to improve robustness against
      structured opponents, which drive the robustness penalty.

  SessionManager — tracks per-seat hand counts and fires session resets
      when both seats have completed SESSION_LENGTH hands.

  play_hand_v2 — game loop for both variants (returns (chain, reward) where
      chain is list of (game_enc_15, stats_7) tuples; identical to v1 but
      importable from here).

  OPPONENT_WEIGHTS  — canonical weight dict (exposed for logging/debugging).
"""

from __future__ import annotations

import random
import sys
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, ROOT)

import torch
from engine.leduc_game import LeducGame, Action
from paper.evaluation.shared.stats_tracker import OpponentStatsTracker


# ── opponent weights ────────────────────────────────────────────────────────────
# CFR and heuristic are the most structured opponents. They dominate the
# robustness penalty because they are the hardest to beat. Oversampling them
# 3× ensures the modulation head spends more time adapting to high-quality play.

OPPONENT_WEIGHTS: dict[str, float] = {
    "cfr":              3.0,
    "heuristic":        3.0,
    "tight_passive":    1.0,
    "tight_aggressive": 1.0,
    "loose_passive":    1.0,
    "loose_aggressive": 1.0,
    "maniac":           1.0,
    "random":           1.0,
}


class WeightedOpponentSampler:
    """
    Samples opponents by name with configurable weights.

    Usage:
        sampler = WeightedOpponentSampler(opponents, OPPONENT_WEIGHTS)
        name, opp = sampler.sample()
    """

    def __init__(self, opponents: dict, weights: dict[str, float] | None = None):
        self.opponents = opponents
        weights = weights or OPPONENT_WEIGHTS
        self.keys = list(opponents.keys())
        self.probs = [weights.get(k, 1.0) for k in self.keys]
        total = sum(self.probs)
        self.probs = [p / total for p in self.probs]

    def sample(self) -> tuple[str, object]:
        key = random.choices(self.keys, weights=self.probs, k=1)[0]
        return key, self.opponents[key]

    def expected_frequencies(self) -> dict[str, float]:
        """Return expected fraction of sessions per opponent (for logging)."""
        return dict(zip(self.keys, self.probs))


class SessionManager:
    """
    Tracks per-seat hand counts. A session ends when both seats have
    completed `session_length` hands. On session end, trackers are reset
    and a new opponent is drawn from the sampler.

    Usage:
        sm = SessionManager(session_length=100, pool_means=..., prior_strength=20.0,
                            sampler=sampler)
        sm.reset()  # sets initial opponent

        # In training loop:
        name, opp = sm.current_opponent()
        chain, reward = play_hand_v2(agent, opp, sm.tracker(learner_id), learner_id)
        sm.record_hand(learner_id)  # increments count, triggers reset if needed
    """

    def __init__(
        self,
        session_length: int,
        pool_means: dict,
        prior_strength: float,
        sampler: WeightedOpponentSampler,
    ):
        self.session_length = session_length
        self.pool_means = pool_means
        self.prior_strength = prior_strength
        self.sampler = sampler

        self._trackers: dict[int, OpponentStatsTracker] = {
            0: OpponentStatsTracker(pool_means, prior_strength, session_length),
            1: OpponentStatsTracker(pool_means, prior_strength, session_length),
        }
        self._counts = {0: 0, 1: 0}
        self._opp_name: str = ""
        self._opp: object = None

        # sample first opponent immediately
        self._sample_new_opponent()

    def _sample_new_opponent(self):
        self._opp_name, self._opp = self.sampler.sample()

    def current_opponent(self) -> tuple[str, object]:
        return self._opp_name, self._opp

    def tracker(self, learner_id: int) -> OpponentStatsTracker:
        return self._trackers[learner_id]

    def record_hand(self, learner_id: int):
        """
        Increment hand count for `learner_id`. If both seats have reached
        `session_length`, reset both trackers and sample a new opponent.
        """
        self._counts[learner_id] += 1
        if (self._counts[0] >= self.session_length and
                self._counts[1] >= self.session_length):
            for t in self._trackers.values():
                t.reset()
            self._counts = {0: 0, 1: 0}
            self._sample_new_opponent()


# ── game loop ───────────────────────────────────────────────────────────────────

# Shared 15-dim game encoding (duplicated from v1 to avoid circular imports).
_CARD_MAP  = {'J': 0, 'Q': 1, 'K': 2}
_MAX_CHIPS = 13


def encode_game_state(obs, viewer_id: int) -> torch.Tensor:
    """15-dim game-state tensor (matches ValueBasedAgent encoding)."""
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


def play_hand_v2(agent, opponent, tracker: OpponentStatsTracker,
                 learner_id: int = 0):
    """
    Play one hand. The learner is `agent` at `learner_id`; opponent at the
    other seat. Returns:
        chain  : list of (game_enc_15, stats_7_copy) for each learner post-action state
        reward : terminal reward for the learner

    Works for both UngatedModAgent and StateGatedModAgent — both expose
    `select_action(obs, opp_stats=stats)`.
    """
    game = LeducGame()
    game.reset()
    chain = []
    prev_raise = False
    prev_round = -1

    while not game.is_finished:
        cp = game.current_player
        obs = game.get_observation(viewer_id=cp)
        if obs.current_round != prev_round:
            prev_raise = False
            prev_round = obs.current_round

        if cp == learner_id:
            stats = tracker.get_features()
            action = agent.select_action(obs, opp_stats=stats)
            post_obs, _ = LeducGame.simulate_action(obs, action)
            game_enc = encode_game_state(post_obs, viewer_id=learner_id)
            chain.append((game_enc, stats.copy()))
        else:
            action = opponent.select_action(obs)
            tracker.update_action(action, obs.current_round,
                                  prev_raise, obs.legal_actions)

        prev_raise = (action == Action.RAISE)
        game.step(action)

    tracker.update_hand_end()
    rewards = game.get_reward()
    return chain, rewards[learner_id]
