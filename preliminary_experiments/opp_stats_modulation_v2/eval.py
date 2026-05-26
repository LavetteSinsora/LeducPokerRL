"""
opp_stats_modulation_v2 — Shared Evaluation Script
====================================================
Loads trained checkpoints for one or both variants and runs a full
stat-aware pool evaluation using comparison_protocol.py.

Outputs a comparison table against:
  - baseline_value_v1
  - opp_stats_modulation_v1/variant_a_td

Usage:
  python eval.py --variant variant_a_ungated
  python eval.py --variant variant_b_state_gated
  python eval.py --all                          # evaluate both variants
  python eval.py --all --rounds 5000            # custom round count
  python eval.py --variant variant_b_state_gated --gate-analysis
      # print per-state gate activations after evaluation
"""

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, ROOT)

import numpy as np
import torch

from paper.evaluation.comparison_protocol import (
    STANDARD_OPPONENT_KEYS,
    build_standard_opponents,
    evaluate_stat_aware_pool,
    format_pool_summary,
    compute_pool_summary,
)
from paper.evaluation.shared.training_recipe import play_hand_v2
from preliminary_experiments.opp_stats_modulation_v2.variant_a_ungated.agent import (
    UngatedModAgent,
)
from preliminary_experiments.opp_stats_modulation_v2.variant_b_state_gated.agent import (
    StateGatedModAgent,
)

# ── constants ──────────────────────────────────────────────────────────────────

SESSION_LENGTH  = 100
PRIOR_STRENGTH  = 20.0
DEFAULT_ROUNDS  = 5_000
OPPONENT_KEYS   = list(STANDARD_OPPONENT_KEYS)

# Known baseline results from prior experiments (for comparison table)
BASELINES = {
    "baseline_value_v1": {
        "heuristic":        +0.295,
        "cfr":              -0.070,
        "tight_passive":    +0.745,
        "tight_aggressive": +0.703,
        "loose_passive":    +0.841,
        "loose_aggressive": +0.623,
        "maniac":           +1.165,
        "random":           +1.353,
        "_avg":             +0.580,
    },
    "v1_variant_a_td": {
        "heuristic":        +0.008,
        "cfr":              -0.194,
        "tight_passive":    +0.704,
        "tight_aggressive": +0.581,
        "loose_passive":    +0.797,
        "loose_aggressive": +0.560,
        "maniac":           +1.533,
        "random":           +1.644,
        "_avg":             +0.539,
    },
}


# ── helpers ───────────────────────────────────────────────────────────────────

def _write_json(path, data):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def _load_pool_means(out_dir: str) -> dict:
    """Load pool priors, falling back to shared augmentation priors."""
    for path in [
        os.path.join(out_dir, "pool_priors.json"),
        os.path.join(ROOT, "preliminary_experiments", "opp_stats_input_aug",
                     "outputs", "pool_random", "pool_priors.json"),
    ]:
        if os.path.exists(path):
            with open(path) as f:
                return json.load(f)
    raise FileNotFoundError(
        "No pool_priors.json found. Run train.py first to generate priors.")


def _print_comparison_table(variant_name: str, scores: dict, summary: dict):
    """Print a side-by-side comparison table vs baselines."""
    header = f"\n{'Opponent':<22} {'baseline_v1':>12} {'v1_a_td':>12} {variant_name:>22}"
    print(header)
    print("-" * len(header))
    for key in OPPONENT_KEYS:
        b1  = BASELINES["baseline_value_v1"].get(key, float("nan"))
        b2  = BASELINES["v1_variant_a_td"].get(key, float("nan"))
        cur = scores.get(key, float("nan"))
        # Mark if current beats both baselines on this opponent
        marker = " ★" if cur > b1 and cur > b2 else ""
        print(f"  {key:<20} {b1:>+12.3f} {b2:>+12.3f} {cur:>+22.3f}{marker}")
    print("-" * len(header))
    b1_avg  = BASELINES["baseline_value_v1"]["_avg"]
    b2_avg  = BASELINES["v1_variant_a_td"]["_avg"]
    cur_avg = summary["avg"]
    cur_rob = summary["robustness"]
    print(f"  {'avg':<20} {b1_avg:>+12.3f} {b2_avg:>+12.3f} {cur_avg:>+22.3f}")
    print(f"  {'robustness':<20} {'—':>12} {'—':>12} {cur_rob:>+22.3f}")
    marker_avg = " ★ beats both baselines" if cur_avg > b1_avg and cur_avg > b2_avg else ""
    print(f"\n  Summary: {format_pool_summary(summary)}{marker_avg}\n")


def _gate_analysis(agent: StateGatedModAgent, opponents: dict, pool_means: dict):
    """
    Print gate activation statistics for each opponent using prototype stats.

    Loads prototype stats from shared_data and evaluates the gate across
    a sample of game-state encodings. Useful for verifying that the gate
    has learned to differentiate state types.
    """
    proto_path = os.path.join(ROOT, "paper", "evaluation", "shared", "data",
                              "opponent_prototype_stats.json")
    if not os.path.exists(proto_path):
        print("  [gate analysis] prototype stats not found, skipping.")
        return

    with open(proto_path) as f:
        proto = json.load(f)

    # Build a diverse sample of game state encodings by simulating random hands
    from engine.leduc_game import LeducGame, Action as Act
    from paper.evaluation.shared.training_recipe import (
        encode_game_state)
    from agents.rule_based.random_agent import RandomAgent  # type: ignore

    rng_agent = RandomAgent()
    agent.set_train_mode(False)
    state_encs = []
    game = LeducGame()
    for _ in range(200):
        game.reset()
        while not game.is_finished:
            cp = game.current_player
            obs = game.get_observation(viewer_id=cp)
            enc = encode_game_state(obs, cp)
            state_encs.append(enc)
            game.step(rng_agent.select_action(obs))

    state_encs = state_encs[:500]  # cap at 500 states

    print("\n  Gate activation analysis (mean gate value per opponent prototype):")
    print(f"  {'Opponent':<22} {'mean_gate':>10} {'std_gate':>10}")
    print("  " + "-" * 42)

    for opp_name in OPPONENT_KEYS:
        if opp_name not in proto:
            continue
        stats_vec = torch.tensor(proto[opp_name], dtype=torch.float32).unsqueeze(0)
        gate_vals = []
        with torch.no_grad():
            for enc in state_encs:
                game_enc = enc.unsqueeze(0)
                gate = agent.get_gate_value(game_enc, stats_vec)
                gate_vals.append(gate)
        mean_g = float(np.mean(gate_vals))
        std_g  = float(np.std(gate_vals))
        print(f"  {opp_name:<22} {mean_g:>+10.4f} {std_g:>10.4f}")


# ── variant evaluators ────────────────────────────────────────────────────────

def evaluate_variant(
    variant: str,
    opponents: dict,
    pool_means: dict,
    num_rounds: int,
    gate_analysis: bool = False,
    checkpoint: str = "checkpoint_best_robust",
) -> dict:
    out_dir = os.path.join(HERE, "outputs", variant)

    if variant == "variant_a_ungated":
        agent = UngatedModAgent()
        ckpt_path = os.path.join(out_dir, f"{checkpoint}.pt")
        if not os.path.exists(ckpt_path):
            ckpt_path = os.path.join(out_dir, "checkpoint.pt")
        agent.load_model(ckpt_path)
    elif variant == "variant_b_state_gated":
        agent = StateGatedModAgent()
        ckpt_path = os.path.join(out_dir, f"{checkpoint}.pt")
        if not os.path.exists(ckpt_path):
            ckpt_path = os.path.join(out_dir, "checkpoint.pt")
        agent.load_model(ckpt_path)
    else:
        raise ValueError(f"Unknown variant: {variant}")

    print(f"\nEvaluating {variant} from {ckpt_path}")
    print(f"  Rounds per opponent: {num_rounds} | Session length: {SESSION_LENGTH}")

    result  = evaluate_stat_aware_pool(
        agent=agent,
        opponents=opponents,
        play_hand_fn=play_hand_v2,
        pool_means=pool_means,
        num_rounds=num_rounds,
        session_length=SESSION_LENGTH,
        prior_strength=PRIOR_STRENGTH,
        opponent_keys=OPPONENT_KEYS,
        alternate_positions=True,
    )
    scores  = result["scores"]
    summary = result["summary"]

    _print_comparison_table(variant, scores, summary)

    # save evaluation.json in the variant output dir
    evaluation = {
        "variant":      variant,
        "checkpoint":   ckpt_path,
        "num_rounds":   num_rounds,
        "scores":       scores,
        "summary": {
            "avg":        round(summary["avg"], 4),
            "worst_case": round(summary["worst_case"], 4),
            "robustness": round(summary["robustness"], 4),
            "std":        round(summary["std"], 4),
        },
        "baselines":    BASELINES,
    }
    eval_path = os.path.join(out_dir, "evaluation.json")
    _write_json(eval_path, evaluation)
    print(f"  Saved to {eval_path}")

    if gate_analysis and isinstance(agent, StateGatedModAgent):
        _gate_analysis(agent, opponents, pool_means)

    return evaluation


# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate opp_stats_modulation_v2 variants")
    parser.add_argument(
        "--variant", choices=["variant_a_ungated", "variant_b_state_gated"],
        help="Which variant to evaluate")
    parser.add_argument(
        "--all", action="store_true",
        help="Evaluate both variants")
    parser.add_argument(
        "--rounds", type=int, default=DEFAULT_ROUNDS,
        help=f"Rounds per opponent (default: {DEFAULT_ROUNDS})")
    parser.add_argument(
        "--checkpoint", default="checkpoint_best_robust",
        help="Checkpoint file stem (default: checkpoint_best_robust)")
    parser.add_argument(
        "--gate-analysis", action="store_true",
        help="Print gate activation analysis for variant_b (if applicable)")
    args = parser.parse_args()

    if not args.variant and not args.all:
        parser.error("Specify --variant or --all")

    variants_to_run = (
        ["variant_a_ungated", "variant_b_state_gated"]
        if args.all else [args.variant]
    )

    # Determine output dir for pool priors (prefer variant_a if both running)
    first_out = os.path.join(HERE, "outputs", variants_to_run[0])
    pool_means = _load_pool_means(first_out)
    opponents  = build_standard_opponents(ROOT)

    all_results = {}
    for variant in variants_to_run:
        out_dir = os.path.join(HERE, "outputs", variant)
        try:
            pm = _load_pool_means(out_dir)
        except FileNotFoundError:
            pm = pool_means

        res = evaluate_variant(
            variant=variant,
            opponents=opponents,
            pool_means=pm,
            num_rounds=args.rounds,
            gate_analysis=args.gate_analysis,
            checkpoint=args.checkpoint,
        )
        all_results[variant] = res

    if args.all:
        print("\n" + "=" * 65)
        print("  SUMMARY COMPARISON")
        print("=" * 65)
        print(f"  {'Metric':<25} {'baseline_v1':>12} {'v1_a_td':>12} "
              f"{'v2a_ungated':>14} {'v2b_gated':>14}")
        print("  " + "-" * 75)
        for key in OPPONENT_KEYS:
            b1  = BASELINES["baseline_value_v1"].get(key, float("nan"))
            b2  = BASELINES["v1_variant_a_td"].get(key, float("nan"))
            va  = all_results.get("variant_a_ungated", {}).get(
                "scores", {}).get(key, float("nan"))
            vb  = all_results.get("variant_b_state_gated", {}).get(
                "scores", {}).get(key, float("nan"))
            print(f"  {key:<25} {b1:>+12.3f} {b2:>+12.3f} {va:>+14.3f} {vb:>+14.3f}")
        print("  " + "-" * 75)
        b1_avg = BASELINES["baseline_value_v1"]["_avg"]
        b2_avg = BASELINES["v1_variant_a_td"]["_avg"]
        va_s = all_results.get("variant_a_ungated", {}).get("summary", {})
        vb_s = all_results.get("variant_b_state_gated", {}).get("summary", {})
        print(f"  {'avg':<25} {b1_avg:>+12.3f} {b2_avg:>+12.3f} "
              f"{va_s.get('avg', float('nan')):>+14.3f} "
              f"{vb_s.get('avg', float('nan')):>+14.3f}")
        print(f"  {'robustness':<25} {'—':>12} {'—':>12} "
              f"{va_s.get('robustness', float('nan')):>+14.3f} "
              f"{vb_s.get('robustness', float('nan')):>+14.3f}")
        print()
