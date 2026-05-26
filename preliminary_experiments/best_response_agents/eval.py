"""
opponent_value_tables_v1 — Final Evaluation Script
===================================================
Loads each per-opponent checkpoint and produces:

  1. Diagonal scores — agent_X vs opponent_X (its training opponent)
     Confirms the value function learned to exploit that specific opponent.

  2. Full cross-eval matrix — agent_X vs all 6 opponents
     Shows how opponent-specific the learned value function is: a well-trained
     per-opponent agent should score significantly better on its training opponent
     than a generic agent would, and may perform differently on other opponents.

Writes outputs/eval_summary.json with the full matrix.

Usage:
    python eval.py               # full 5000-round eval
    python eval.py --rounds 1000 # faster version
"""

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, ROOT)

from agents.value_based.agent import ValueBasedAgent
from agents.evaluation import quick_evaluate
from agents.rule_based import ALL_AGENTS

OUT_ROOT    = os.path.join(HERE, "outputs")
AGENT_KEYS  = list(ALL_AGENTS.keys())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds", type=int, default=5000,
                        help="Rounds per matchup (default: 5000)")
    args = parser.parse_args()

    # ── Build fixed opponent pool ──────────────────────────────────────────
    opponents = {k: v() for k, v in ALL_AGENTS.items()}
    for opp in opponents.values():
        opp.set_train_mode(False)

    # ── Load all trained agents ────────────────────────────────────────────
    agents  = {}
    missing = []
    for key in AGENT_KEYS:
        best_ckpt  = os.path.join(OUT_ROOT, key, "checkpoint_best.pt")
        final_ckpt = os.path.join(OUT_ROOT, key, "checkpoint.pt")
        ckpt = best_ckpt if os.path.exists(best_ckpt) else final_ckpt

        if not os.path.exists(ckpt):
            print(f"  WARNING: no checkpoint for {key} — skipping")
            missing.append(key)
            continue

        agent = ValueBasedAgent()
        agent.load_model(ckpt)
        agent.set_train_mode(False)
        agents[key] = agent
        src = "best" if ckpt == best_ckpt else "final"
        print(f"  Loaded {key:<20s} ({src})")

    if not agents:
        print("\nNo checkpoints found. Run train.py first.")
        sys.exit(1)

    total_matchups = len(agents) * len(opponents)
    print(f"\nEvaluating {len(agents)} agents × {len(opponents)} opponents × "
          f"{args.rounds} rounds  ({total_matchups} matchups total)\n")

    # ── Full cross-eval matrix ─────────────────────────────────────────────
    # matrix[agent_key][opp_key] = avg_chips_per_round
    matrix = {}
    for agent_key, agent in agents.items():
        matrix[agent_key] = {}
        row_parts = []
        for opp_key, opp in opponents.items():
            score = round(quick_evaluate(agent, opp, num_rounds=args.rounds), 4)
            matrix[agent_key][opp_key] = score
            marker = "*" if agent_key == opp_key else " "
            row_parts.append(f"{opp_key[:8]:8s}:{score:+.3f}{marker}")
        print(f"  agent[{agent_key:<18s}]  " + "  ".join(row_parts))

    # ── Diagonal (training-opponent) summary ───────────────────────────────
    print("\n── Diagonal: trained-opponent scores ──")
    diagonal = {}
    for key in agents:
        if key in matrix.get(key, {}):
            s = matrix[key][key]
            diagonal[key] = s
            print(f"  {key:<20s}  {s:+.4f} chips/round")

    # ── Write summary ──────────────────────────────────────────────────────
    summary = {
        "eval_rounds":    args.rounds,
        "agents_loaded":  list(agents.keys()),
        "missing_agents": missing,
        "matrix":         matrix,
        "diagonal":       diagonal,
        "notes": (
            "matrix[agent][opp] = avg chips/round for the agent trained against "
            "{agent} when playing against {opp}. Diagonal entries (*) are the "
            "training-opponent matchups and represent the primary quality metric."
        ),
    }

    out_path = os.path.join(OUT_ROOT, "eval_summary.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\neval_summary.json written → {out_path}")


if __name__ == "__main__":
    main()
