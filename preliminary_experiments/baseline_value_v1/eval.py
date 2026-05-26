"""
baseline_value_v1 — Final Evaluation Script
=============================================
Loads checkpoint_best.pt and runs a comprehensive 5000-round evaluation
against all 8 opponents. Writes evaluation.json and prints a summary.

Usage:
    python eval.py
    python eval.py --checkpoint outputs/checkpoint.pt
"""

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, ROOT)

from agents.value_based.agent import ValueBasedAgent
from agents.heuristic.agent import HeuristicAgent
from agents.cfr.agent import CFRAgent
from agents.evaluation import compute_robustness_metrics, evaluate_agents

from agents.rule_based.tight_passive    import TightPassiveAgent
from agents.rule_based.tight_aggressive  import TightAggressiveAgent
from agents.rule_based.loose_passive    import LoosePassiveAgent
from agents.rule_based.loose_aggressive  import LooseAggressiveAgent
from agents.rule_based.maniac           import ManiacAgent
from agents.rule_based.random_agent     import RandomAgent

OUT = os.path.join(HERE, "outputs")
EVAL_ROUNDS = 5000


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=os.path.join(OUT, "checkpoint_best.pt"))
    args = parser.parse_args()

    if not os.path.exists(args.checkpoint):
        print(f"Checkpoint not found: {args.checkpoint}")
        sys.exit(1)

    print(f"\n{'='*65}")
    print(f"  baseline_value_v1 — Final Evaluation")
    print(f"  checkpoint : {os.path.basename(args.checkpoint)}")
    print(f"  rounds     : {EVAL_ROUNDS:,} per opponent  (8 opponents)")
    print(f"{'='*65}\n")

    agent = ValueBasedAgent(model_path=args.checkpoint)
    agent.set_train_mode(False)

    cfr_path = os.path.join(ROOT, "agents", "cfr", "checkpoint.pt")
    opponents = {
        "heuristic":        HeuristicAgent(),
        "cfr":              CFRAgent(model_path=cfr_path),
        "tight_passive":    TightPassiveAgent(),
        "tight_aggressive": TightAggressiveAgent(),
        "loose_passive":    LoosePassiveAgent(),
        "loose_aggressive": LooseAggressiveAgent(),
        "maniac":           ManiacAgent(),
        "random":           RandomAgent(),
    }

    results = {}
    for name, opp in opponents.items():
        opp.set_train_mode(False)
        print(f"  Evaluating vs {name} ...")
        r = evaluate_agents(agent, opp, num_rounds=EVAL_ROUNDS)
        results[name] = {
            "avg_chips_per_round": round(r.agent_0_avg_chips, 4),
            "total_chips":         round(r.agent_0_total_chips, 2),
            "rounds":              r.num_rounds,
        }
        print(f"    → {r.agent_0_avg_chips:+.4f} chips/round")

    evaluation = {
        "experiment_id": "baseline_value_v1",
        "checkpoint":    os.path.basename(args.checkpoint),
        "eval_rounds":   EVAL_ROUNDS,
        "scores":        {k: v["avg_chips_per_round"] for k, v in results.items()},
        **compute_robustness_metrics({k: v["avg_chips_per_round"] for k, v in results.items()}),
        "details":       results,
    }
    eval_path = os.path.join(OUT, "evaluation.json")
    with open(eval_path, "w") as f:
        json.dump(evaluation, f, indent=2)

    print(f"\n{'─'*50}")
    print(f"  {'Opponent':<20}  {'Avg chips/round':>16}")
    print(f"  {'─'*38}")
    for name, v in results.items():
        print(f"  {name:<20}  {v['avg_chips_per_round']:>+16.4f}")
    print(f"{'─'*50}")
    print(f"  {'Overall avg':<20}  {evaluation['avg']:>+16.4f}")
    print(f"  {'Worst case':<20}  {evaluation['worst_case']:>+16.4f}")
    print(f"  {'Robustness':<20}  {evaluation['robustness']:>+16.4f}")
    print(f"\n  evaluation.json saved to {eval_path}")


if __name__ == "__main__":
    main()
