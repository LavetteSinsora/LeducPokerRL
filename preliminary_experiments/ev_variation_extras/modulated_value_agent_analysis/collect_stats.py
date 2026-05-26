"""
collect_stats.py — Collect 4-dim OpponentStats for each rule-based agent.

Runs 10,000 hands of ModulatedValueAgent (player 0) vs each rule-based agent
(player 1) using PokerSession, which automatically accumulates OpponentStats
from the perspective of player 0.

After 10,000 hands the stats are fully saturated (confidence = 1.0 since
saturation threshold is 50 hands).

Output:
    modulated_value_agent_analysis/opponent_stats.json
    {
      "tight_passive": [fold_rate, raise_rate, fold_to_raise_rate, confidence],
      ...
    }

Run from project root:
    python -m preliminary_experiments.ev_variation_extras.modulated_value_agent_analysis.collect_stats
"""

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, ROOT)

from engine.poker_session import PokerSession
from preliminary_experiments.promoted_registry.modulated_value.agent import ModulatedValueAgent
from agents.rule_based import ALL_AGENTS

NUM_HANDS   = 10_000
CKPT_PATH   = os.path.join(ROOT, "agents", "modulated_value", "checkpoint.pt")
OUTPUT_PATH = os.path.join(HERE, "opponent_stats.json")
RULE_BASED_KEYS = [
    "tight_passive",
    "tight_aggressive",
    "loose_passive",
    "loose_aggressive",
    "maniac",
    "random",
]


def collect_stats_for_opponent(mod_agent, opp_agent, num_hands: int) -> list:
    """
    Play num_hands between mod_agent (player 0) and opp_agent (player 1).
    Returns the 4-dim stats vector that player 0 accumulated about player 1.
    """
    session = PokerSession()

    for _ in range(num_hands):
        session.new_hand()
        while not session.is_finished:
            cp  = session.current_player
            obs = session.get_observation(viewer_id=cp)
            if cp == 0:
                action = mod_agent.select_action(obs)
            else:
                action = opp_agent.select_action(obs)
            session.step(action)

    # stats[0] = what player 0 observed about player 1 (the rule-based agent)
    return session.stats[0].to_feature_vector()


def main():
    print(f"Loading ModulatedValueAgent from {CKPT_PATH}")
    mod_agent = ModulatedValueAgent(model_path=CKPT_PATH)
    mod_agent.set_train_mode(False)

    results = {}
    for key in RULE_BASED_KEYS:
        opp_agent = ALL_AGENTS[key]()
        opp_agent.set_train_mode(False)
        print(f"  Running {NUM_HANDS:,} hands vs {key} ...", end=" ", flush=True)
        stats_vec = collect_stats_for_opponent(mod_agent, opp_agent, NUM_HANDS)
        results[key] = stats_vec
        fold_r, raise_r, ftr_r, conf = stats_vec
        print(f"fold={fold_r:.3f}  raise={raise_r:.3f}  ftr={ftr_r:.3f}  conf={conf:.3f}")

    output = {
        "_meta": {
            "description": (
                "4-dim OpponentStats (fold_rate, raise_rate, fold_to_raise_rate, confidence) "
                "collected by ModulatedValueAgent playing 10,000 hands against each rule-based agent."
            ),
            "num_hands_per_opponent": NUM_HANDS,
            "modulated_agent_checkpoint": CKPT_PATH,
            "feature_order": [
                "fold_rate",
                "raise_rate",
                "fold_to_raise_rate",
                "confidence",
            ],
            "confidence_formula": "min(hands_observed / 50.0, 1.0)",
            "note": (
                "This is the simpler 4-dim OpponentStats from engine/poker_session.py, "
                "NOT the 7-dim OpponentStatsTracker used in shared_data/opponent_prototype_stats.json."
            ),
        },
    }
    output.update(results)

    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nSaved opponent stats → {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
