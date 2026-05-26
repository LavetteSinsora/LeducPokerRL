"""
opp_stats_input_augmentation_v1 — Final Evaluation Script
========================================
Loads checkpoint_best.pt for a given variant and runs a comprehensive
5000-round evaluation against all 8 opponents.

Stats accumulate over the 5000 rounds per opponent (no reset), simulating
how the agent adapts as it observes more of the opponent's behavior.

Usage:
    python eval.py --variant self_play
    python eval.py --variant pool_random
    python eval.py --variant pool_seq
    python eval.py --variant self_play --checkpoint outputs/self_play/checkpoint.pt
"""

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, ROOT)

from paper.evaluation.comparison_protocol import (
    STANDARD_OPPONENT_KEYS,
    build_standard_opponents,
    evaluate_stat_aware_pool,
    format_pool_summary,
)

from paper.evaluation.shared.stats_tracker import play_hand
from preliminary_experiments.opp_stats_input_aug.agent import StatAugValueAgent

EVAL_ROUNDS     = 5000
PRIOR_STRENGTH  = 20.0
SESSION_LENGTH  = 100
OPPONENT_KEYS   = list(STANDARD_OPPONENT_KEYS)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant",    required=True,
                        choices=["self_play", "pool_random", "pool_seq"])
    parser.add_argument("--checkpoint", default=None,
                        help="Override checkpoint path")
    args = parser.parse_args()

    out_dir = os.path.join(HERE, "outputs", args.variant)
    ckpt    = args.checkpoint or os.path.join(out_dir, "checkpoint_best.pt")

    if not os.path.exists(ckpt):
        print(f"Checkpoint not found: {ckpt}")
        sys.exit(1)

    priors_path = os.path.join(out_dir, "pool_priors.json")
    if not os.path.exists(priors_path):
        print(f"pool_priors.json not found in {out_dir}. Run train.py first.")
        sys.exit(1)
    pool_means = json.load(open(priors_path))

    print(f"\n{'='*65}")
    print(f"  opp_stats_input_augmentation_v1 — Final Evaluation  [{args.variant}]")
    print(f"  checkpoint : {os.path.basename(ckpt)}")
    print(f"  rounds     : {EVAL_ROUNDS:,} per opponent  (both seats, 100-hand resets)")
    print(f"{'='*65}\n")

    agent = StatAugValueAgent(model_path=ckpt)
    agent.set_train_mode(False)

    opponents = build_standard_opponents(ROOT)
    pool_eval = evaluate_stat_aware_pool(
        agent=agent,
        opponents=opponents,
        play_hand_fn=play_hand,
        pool_means=pool_means,
        num_rounds=EVAL_ROUNDS,
        session_length=SESSION_LENGTH,
        prior_strength=PRIOR_STRENGTH,
        opponent_keys=OPPONENT_KEYS,
        alternate_positions=True,
    )
    results = pool_eval["details"]
    for name in OPPONENT_KEYS:
        print(f"  {name:<22}  {results[name]['avg_chips_per_round']:+.4f} chips/round")

    evaluation = {
        "experiment_id": f"opp_stats_input_augmentation_v1_{args.variant}",
        "variant":       args.variant,
        "checkpoint":    os.path.basename(ckpt),
        "eval_rounds":   EVAL_ROUNDS,
        "scores":  {k: v["avg_chips_per_round"] for k, v in results.items()},
        "overall_avg": pool_eval["summary"]["avg"],
        "worst_case": pool_eval["summary"]["worst_case"],
        "best_case": pool_eval["summary"]["best_case"],
        "robustness": pool_eval["summary"]["robustness"],
        "details": results,
    }
    eval_path = os.path.join(out_dir, "evaluation.json")
    with open(eval_path, "w") as f:
        json.dump(evaluation, f, indent=2)

    print(f"\n{'─'*50}")
    print(f"  {format_pool_summary(pool_eval['summary'])}")
    print(f"  evaluation.json → {eval_path}")


if __name__ == "__main__":
    main()
