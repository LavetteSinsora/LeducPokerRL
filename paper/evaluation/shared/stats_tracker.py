"""
OpponentStatsTracker
=====================
Tracks 6 round-stratified opponent statistics using Beta-Bernoulli smoothing
toward pool-average priors. Confidence is derived from hands seen vs prior
strength, replacing the need for a separate normalization scheme.

Features (7 total — 6 rates + confidence):
  1. preflop_fold_rate       — fold freq on pre-flop
  2. preflop_raise_rate      — raise freq on pre-flop
  3. flop_raise_rate         — raise freq on flop  (spike vs preflop = pairing signal)
  4. preflop_fold_to_raise   — fold freq when facing a pre-flop raise
  5. flop_fold_to_raise      — fold freq when facing a flop raise
  6. raise_after_raise_rate  — re-raise freq when facing any raise (maniac signal)
  7. confidence              — n / (n + S), how much data has shifted from prior

Posterior mean: p̂ = (k + α) / (n + α + β)
  where α = pool_mean × S, β = (1 - pool_mean) × S, S = prior_strength
"""

import numpy as np
from engine.leduc_game import LeducGame, Action

STAT_KEYS = [
    "preflop_fold_rate",
    "preflop_raise_rate",
    "flop_raise_rate",
    "preflop_fold_to_raise",
    "flop_fold_to_raise",
    "raise_after_raise_rate",
]
N_STATS = len(STAT_KEYS) + 1   # 6 rates + confidence = 7


class OpponentStatsTracker:
    def __init__(self, pool_means: dict, prior_strength: float = 20.0,
                 session_length: int = 100):
        self.pool_means     = {k: pool_means.get(k, 0.5) for k in STAT_KEYS}
        self.S              = prior_strength
        self.session_length = session_length
        self._k = {s: 0 for s in STAT_KEYS}   # numerators
        self._n = {s: 0 for s in STAT_KEYS}   # denominators
        self.hands_seen = 0

    def update_action(self, action: Action, round_num: int,
                      facing_raise: bool, legal_actions: list):
        """Record one observed opponent action."""
        if round_num == 0:                          # pre-flop
            self._n["preflop_fold_rate"]  += 1
            self._n["preflop_raise_rate"] += 1
            if action == Action.FOLD:
                self._k["preflop_fold_rate"]  += 1
            elif action == Action.RAISE:
                self._k["preflop_raise_rate"] += 1
            if facing_raise:
                self._n["preflop_fold_to_raise"] += 1
                if action == Action.FOLD:
                    self._k["preflop_fold_to_raise"] += 1
        elif round_num == 1:                        # flop
            self._n["flop_raise_rate"] += 1
            if action == Action.RAISE:
                self._k["flop_raise_rate"] += 1
            if facing_raise:
                self._n["flop_fold_to_raise"] += 1
                if action == Action.FOLD:
                    self._k["flop_fold_to_raise"] += 1

        # raise_after_raise: either round, only when re-raise is legal
        if facing_raise and Action.RAISE in legal_actions:
            self._n["raise_after_raise_rate"] += 1
            if action == Action.RAISE:
                self._k["raise_after_raise_rate"] += 1

    def update_hand_end(self):
        """Call once per completed hand."""
        self.hands_seen += 1

    def get_features(self) -> np.ndarray:
        """Return 7-dim float32 array: [6 posterior means, confidence]."""
        feats = []
        for stat in STAT_KEYS:
            k  = self._k[stat]
            n  = self._n[stat]
            pm = self.pool_means[stat]
            alpha = pm * self.S
            beta  = (1.0 - pm) * self.S
            denom = n + alpha + beta
            feats.append((k + alpha) / denom if denom > 0 else pm)
        confidence = self.hands_seen / (self.hands_seen + self.S)
        feats.append(confidence)
        return np.array(feats, dtype=np.float32)

    def reset(self):
        """Reset per-session counts. Prior remains unchanged."""
        self._k = {s: 0 for s in STAT_KEYS}
        self._n = {s: 0 for s in STAT_KEYS}
        self.hands_seen = 0


# ── pool calibration ──────────────────────────────────────────────────────────

def compute_pool_means(opponents: dict, calibration_hands: int = 500,
                       verbose: bool = True) -> dict:
    """
    Play calibration_hands against each opponent using a RandomAgent learner
    (so all action contexts are covered), then average stats across the pool.
    """
    import sys, os
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
    from agents.rule_based.random_agent import RandomAgent

    total_k = {s: 0 for s in STAT_KEYS}
    total_n = {s: 0 for s in STAT_KEYS}
    learner = RandomAgent()

    for name, opp in opponents.items():
        if verbose:
            print(f"  Calibrating vs {name} ({calibration_hands} hands)...", end=" ")
        opp.set_train_mode(False)
        game = LeducGame()
        k = {s: 0 for s in STAT_KEYS}
        n = {s: 0 for s in STAT_KEYS}

        for _ in range(calibration_hands):
            game.reset()
            prev_raise = False
            prev_round = -1
            while not game.is_finished:
                cp = game.current_player
                obs = game.get_observation(viewer_id=cp)
                if obs.current_round != prev_round:
                    prev_raise = False
                    prev_round = obs.current_round
                if cp == 0:
                    action = learner.select_action(obs)
                    prev_raise = (action == Action.RAISE)
                else:
                    action = opp.select_action(obs)
                    _update_raw(k, n, action, obs.current_round, prev_raise, obs.legal_actions)
                game.step(action)

        for s in STAT_KEYS:
            total_k[s] += k[s]
            total_n[s] += n[s]
        if verbose:
            print("done")

    pool_means = {
        s: round(total_k[s] / total_n[s], 4) if total_n[s] > 0 else 0.5
        for s in STAT_KEYS
    }
    if verbose:
        print(f"  Pool means: { {k: f'{v:.3f}' for k, v in pool_means.items()} }")
    return pool_means


def _update_raw(k, n, action, round_num, facing_raise, legal_actions):
    if round_num == 0:
        n["preflop_fold_rate"]  += 1
        n["preflop_raise_rate"] += 1
        if action == Action.FOLD:
            k["preflop_fold_rate"]  += 1
        elif action == Action.RAISE:
            k["preflop_raise_rate"] += 1
        if facing_raise:
            n["preflop_fold_to_raise"] += 1
            if action == Action.FOLD:
                k["preflop_fold_to_raise"] += 1
    elif round_num == 1:
        n["flop_raise_rate"] += 1
        if action == Action.RAISE:
            k["flop_raise_rate"] += 1
        if facing_raise:
            n["flop_fold_to_raise"] += 1
            if action == Action.FOLD:
                k["flop_fold_to_raise"] += 1
    if facing_raise and Action.RAISE in legal_actions:
        n["raise_after_raise_rate"] += 1
        if action == Action.RAISE:
            k["raise_after_raise_rate"] += 1


# ── game loop helper ──────────────────────────────────────────────────────────

def play_hand(agent, opponent, tracker: OpponentStatsTracker,
              learner_id: int = 0):
    """
    Play one hand. The learner is `agent` at position `learner_id`;
    the opponent sits at `1 - learner_id`. Tracker observes opponent actions.

    Returns:
        chain  : list of encoded post-action tensors for the learner
        reward : scalar terminal reward for the learner
    """
    game = LeducGame()
    game.reset()
    chain = []
    prev_raise = False
    prev_round = -1
    opponent_id = 1 - learner_id

    while not game.is_finished:
        cp = game.current_player
        obs = game.get_observation(viewer_id=cp)
        if obs.current_round != prev_round:
            prev_raise = False
            prev_round = obs.current_round

        if cp == learner_id:
            stats = tracker.get_features()
            action = agent.select_action(obs, opp_stats=stats)
            from engine.leduc_game import LeducGame as LG
            post_obs, _ = LG.simulate_action(obs, action)
            encoded = agent.encode_observation(post_obs,
                                               viewer_id=learner_id,
                                               opp_stats=stats)
            chain.append(encoded)
        else:
            action = opponent.select_action(obs)
            tracker.update_action(action, obs.current_round,
                                  prev_raise, obs.legal_actions)

        prev_raise = (action == Action.RAISE)
        game.step(action)

    tracker.update_hand_end()
    rewards = game.get_reward()
    return chain, rewards[learner_id]
