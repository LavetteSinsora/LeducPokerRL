"""
Generate Opponent Prototype Statistics
========================================
Produces opponent_prototype_stats.json — the canonical 7-dimensional behavioral
embedding for each opponent archetype in the opponent pool.

What this produces:
-------------------
For each opponent agent, we play CALIBRATION_HANDS hands against it using a
RandomAgent learner (so all action contexts are covered) and record the
Beta-Bernoulli posterior means for 6 round-stratified behavioral rates, plus
a confidence value.

The 7 features (in order):
  [0] preflop_fold_rate       — how often the opponent folds pre-flop
  [1] preflop_raise_rate      — how often the opponent raises pre-flop
  [2] flop_raise_rate         — how often the opponent raises on the flop
  [3] preflop_fold_to_raise   — how often the opponent folds when facing a pre-flop raise
  [4] flop_fold_to_raise      — how often the opponent folds when facing a flop raise
  [5] raise_after_raise_rate  — how often the opponent re-raises when re-raising is legal
  [6] confidence              — n / (n + S), reflecting how much data shifted the prior

Smoothing formula (Beta-Bernoulli posterior):
  p_hat = (k + alpha) / (n + alpha + beta)
  alpha = pool_mean * S,  beta = (1 - pool_mean) * S,  S = prior_strength = 20

  At 0 hands: p_hat = pool_mean (prior center)
  At 500 hands with S=20: confidence = 500/520 ≈ 0.962 (prior almost entirely replaced by data)

Key distinction from pool-mean prior:
  - pool_mean prior: single average of all opponents' stats, used as Bayesian prior center
    in OpponentStatsTracker when no per-opponent data is available yet.
  - prototype stats (this file): per-opponent characteristic stats after 500 hands of
    calibration. Used as fixed embeddings to represent each opponent archetype.

Usage:
    python generate_prototype_stats.py            # full 500-hand calibration
    python generate_prototype_stats.py --smoke    # 50-hand quick check

Output:
    paper/evaluation/shared/data/opponent_prototype_stats.json  (this directory)
"""

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, ROOT)

import argparse
from engine.leduc_game import LeducGame, Action
from paper.evaluation.shared.stats_tracker import (
    OpponentStatsTracker, STAT_KEYS,
)
from agents.rule_based.random_agent      import RandomAgent
from agents.rule_based.tight_passive     import TightPassiveAgent
from agents.rule_based.tight_aggressive   import TightAggressiveAgent
from agents.rule_based.loose_passive     import LoosePassiveAgent
from agents.rule_based.loose_aggressive   import LooseAggressiveAgent
from agents.rule_based.maniac            import ManiacAgent
from agents.heuristic.agent import HeuristicAgent
from agents.cfr.agent       import CFRAgent
from agents.value_based.agent import ValueBasedAgent

PRIOR_STRENGTH    = 20.0
CALIBRATION_HANDS = 500   # intended full run


def build_opponents():
    cfr_ckpt = os.path.join(ROOT, "agents", "cfr", "checkpoint.pt")
    vb_ckpt  = os.path.join(ROOT, "agents", "value_based", "checkpoint.pt")
    pool = {
        "cfr":              CFRAgent(model_path=cfr_ckpt),
        "value_based":      ValueBasedAgent(model_path=vb_ckpt),
        "tight_passive":    TightPassiveAgent(),
        "tight_aggressive": TightAggressiveAgent(),
        "loose_passive":    LoosePassiveAgent(),
        "loose_aggressive": LooseAggressiveAgent(),
        "maniac":           ManiacAgent(),
        "random":           RandomAgent(),
        "heuristic":        HeuristicAgent(),
    }
    for opp in pool.values():
        opp.set_train_mode(False)
    return pool


def compute_prototype_stats(opponents, calibration_hands, verbose=True):
    """
    Play calibration_hands against each opponent using a RandomAgent learner.
    Returns dict[opp_name -> list[float]] of 7-dim stat vectors.
    """
    # neutral prior for collection (we want raw posterior means, not biased toward any pool)
    neutral_pool = {k: 0.5 for k in STAT_KEYS}
    learner = RandomAgent()
    game    = LeducGame()
    prototypes = {}

    for name, opp in opponents.items():
        tracker = OpponentStatsTracker(
            neutral_pool, PRIOR_STRENGTH, session_length=calibration_hands + 1
        )
        prev_raise = False
        prev_round = -1

        for _ in range(calibration_hands):
            game.reset()
            prev_raise = False
            prev_round = -1
            while not game.is_finished:
                cp  = game.current_player
                obs = game.get_observation(viewer_id=cp)
                if obs.current_round != prev_round:
                    prev_raise = False
                    prev_round = obs.current_round

                if cp == 0:                        # learner seat
                    action = learner.select_action(obs)
                else:                              # opponent seat
                    action = opp.select_action(obs)
                    tracker.update_action(action, obs.current_round,
                                          prev_raise, obs.legal_actions)
                prev_raise = (action == Action.RAISE)
                game.step(action)
            tracker.update_hand_end()

        stats = tracker.get_features().tolist()
        prototypes[name] = stats
        if verbose:
            print(f"  {name:<22}: {[round(v, 4) for v in stats]}")

    return prototypes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true",
                        help="Use 50 hands per opponent instead of 500")
    args = parser.parse_args()

    cal = 50 if args.smoke else CALIBRATION_HANDS
    out_path = os.path.join(HERE, "opponent_prototype_stats.json")

    print(f"Computing opponent prototype stats ({cal} hands per opponent)...")
    opponents  = build_opponents()
    prototypes = compute_prototype_stats(opponents, cal)

    output = {
        "_meta": {
            "description": (
                "Per-opponent prototype behavioral statistics — 7-dimensional embeddings "
                "for each opponent archetype in the opponent pool."
            ),
            "produced_by": "paper/evaluation/shared/data/generate_prototype_stats.py",
            "calibration_hands_per_opponent": cal,
            "prior_strength_S": PRIOR_STRENGTH,
            "learner": "RandomAgent (uniform random actions)",
            "feature_order": [
                "preflop_fold_rate",
                "preflop_raise_rate",
                "flop_raise_rate",
                "preflop_fold_to_raise",
                "flop_fold_to_raise",
                "raise_after_raise_rate",
                "confidence",
            ],
            "smoothing": (
                "Beta-Bernoulli: p_hat = (k + alpha) / (n + alpha + beta), "
                f"alpha = 0.5 * {PRIOR_STRENGTH}, beta = 0.5 * {PRIOR_STRENGTH} "
                "(neutral prior, all rates centered at 0.5 before calibration)"
            ),
            "expected_confidence": round(cal / (cal + PRIOR_STRENGTH), 4),
            "note": (
                "These are per-opponent archetypes, NOT the pool-mean prior. "
                "The pool-mean prior (used in OpponentStatsTracker cold-start) is "
                "a single cross-opponent average. These prototype stats represent each "
                "opponent's characteristic behavior after sufficient observation."
            ),
        },
    }
    output.update(prototypes)

    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    confidence = cal / (cal + PRIOR_STRENGTH)
    print(f"\nSaved {len(prototypes)} opponent prototypes → {out_path}")
    print(f"  confidence level: {confidence:.4f}  ({cal} hands / ({cal} + {PRIOR_STRENGTH}))")
    if args.smoke:
        print("  [SMOKE] Re-run without --smoke for production-quality 500-hand calibration.")


if __name__ == "__main__":
    main()
