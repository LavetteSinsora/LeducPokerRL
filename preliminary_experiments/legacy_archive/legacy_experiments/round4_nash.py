"""
Round 4 Direction 2: Nash Value Network Experiment

Trains a neural value network on exact game-theoretic values from CFR,
then evaluates it against the full agent roster.

Key question: Does training on exact Nash values (zero noise, zero
non-stationarity) produce better tournament performance than TD(0)
self-play trained on noisy experience?

Pipeline:
  1. Run CFR for 10,000 iterations to compute Nash equilibrium
  2. Extract counterfactual values for all ~288 information sets
  3. Train ValueNetwork(15->64->64->1) via supervised MSE regression
  4. Evaluate against 6 opponents (500 rounds each, both positions)
  5. Diagnose: net approximation quality, Nash vs self-play value comparison
  6. Save results + model
"""

import json
import os
import sys
import time
import math

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np

from src.agents.nash_value import NashValueAgent
from src.agents.heuristic import HeuristicAgent
from src.agents.value_based import ValueBasedAgent
from src.agents.cfr_agent import CFRAgent
from src.training.nash_trainer import NashTrainer
from src.training.evaluation import evaluate_agents, compute_robustness_metrics
from src.engine.leduc_game import LeducGame


# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────

CFR_ITERATIONS = 10000
TRAINING_EPOCHS = 10000
LEARNING_RATE = 1e-3
EVAL_ROUNDS = 500  # per matchup, per position
MODEL_PATH = "models/nash_value_agent.pt"
RESULTS_PATH = "experiments/round4_nash_results.json"

# Opponents to evaluate against
OPPONENTS = {
    "heuristic": {"class": "HeuristicAgent", "model_path": None},
    "value_based": {"class": "ValueBasedAgent", "model_path": "models/value_based_agent.pt"},
    "adaptive_value": {"class": "AdaptiveValueAgent", "model_path": "models/adaptive_value_agent.pt"},
    "modulated_value": {"class": "ModulatedValueAgent", "model_path": "models/modulated_value_agent.pt"},
    "entropy_ac": {"class": "EntropyACAgent", "model_path": "models/entropy_ac_agent.pt"},
    "cfr": {"class": "CFRAgent", "model_path": "models/cfr_agent.pt"},
}


# ──────────────────────────────────────────────
# Helper: Load opponent agent
# ──────────────────────────────────────────────

def load_opponent(name: str, config: dict):
    """Load an opponent agent by name."""
    from src.agents.registry import registry

    if name == "heuristic":
        return HeuristicAgent()

    agent = registry.create(name)
    model_path = config.get("model_path")
    if model_path and os.path.exists(model_path):
        agent.load_model(model_path)
        if hasattr(agent, "set_train_mode"):
            agent.set_train_mode(False)
    else:
        print(f"  WARNING: Model not found at {model_path}, using untrained {name}")

    return agent


# ──────────────────────────────────────────────
# Manual evaluation (as specified in task)
# ──────────────────────────────────────────────

def manual_evaluate(agent_a, agent_b, num_rounds: int = 500) -> dict:
    """Evaluate two agents using the exact pattern from the task spec.

    Plays num_rounds with agent_a as P0, then num_rounds with agent_a as P1.
    Returns average chips/round for agent_a.
    """
    game = LeducGame()
    total_reward_a = 0.0

    # agent_a as P0
    for _ in range(num_rounds):
        game.reset()
        agents = {0: agent_a, 1: agent_b}
        while not game.is_finished:
            cp = game.current_player
            obs = game.get_observation(viewer_id=cp)
            action = agents[cp].select_action(obs)
            game.step(action)
        rewards = game.get_reward()
        total_reward_a += rewards[0]

    # agent_a as P1
    for _ in range(num_rounds):
        game.reset()
        agents = {0: agent_b, 1: agent_a}
        while not game.is_finished:
            cp = game.current_player
            obs = game.get_observation(viewer_id=cp)
            action = agents[cp].select_action(obs)
            game.step(action)
        rewards = game.get_reward()
        total_reward_a += rewards[1]

    total_games = 2 * num_rounds
    avg_chips = total_reward_a / total_games
    return {
        "total_reward": total_reward_a,
        "total_games": total_games,
        "avg_chips_per_round": round(avg_chips, 4),
    }


# ──────────────────────────────────────────────
# Diagnostics
# ──────────────────────────────────────────────

def diagnose_approximation(trainer: NashTrainer) -> dict:
    """How well does the neural network approximate the CFR values?"""
    trainer.agent.model.eval()
    X, y = trainer.X, trainer.y

    with torch.no_grad():
        predictions = trainer.agent.model(X)

    errors = (predictions - y).squeeze()
    mse = (errors ** 2).mean().item()
    mae = errors.abs().mean().item()
    max_error = errors.abs().max().item()

    # Per-infoset errors for top-10
    sorted_errors = []
    for i, key in enumerate(trainer.dataset_keys):
        sorted_errors.append({
            "key": key,
            "true": round(y[i].item(), 6),
            "pred": round(predictions[i].item(), 6),
            "error": round(abs(errors[i].item()), 6),
        })
    sorted_errors.sort(key=lambda x: -x["error"])

    return {
        "mse": round(mse, 8),
        "mae": round(mae, 6),
        "max_error": round(max_error, 6),
        "worst_10": sorted_errors[:10],
        "best_10": sorted_errors[-10:],
    }


def diagnose_nash_vs_selfplay(trainer: NashTrainer) -> dict:
    """Compare Nash values vs self-play (ValueBasedAgent) values."""
    vba_path = "models/value_based_agent.pt"
    if not os.path.exists(vba_path):
        print("  WARNING: value_based_agent.pt not found, skipping comparison")
        return {"skipped": True, "reason": "model not found"}

    vba = ValueBasedAgent(model_path=vba_path)
    vba.model.eval()

    comparison = trainer.get_value_comparison(vba)

    # Summary statistics
    divergences = [c["divergence"] for c in comparison]
    avg_div = sum(divergences) / len(divergences) if divergences else 0
    max_div = max(divergences) if divergences else 0

    # Categorize: where does Nash differ most from self-play?
    preflop_divs = [c for c in comparison if ":" in c["key"] and "/" not in c["key"]]
    flop_divs = [c for c in comparison if "/" in c["key"]]

    avg_preflop_div = sum(c["divergence"] for c in preflop_divs) / len(preflop_divs) if preflop_divs else 0
    avg_flop_div = sum(c["divergence"] for c in flop_divs) / len(flop_divs) if flop_divs else 0

    return {
        "avg_divergence": round(avg_div, 6),
        "max_divergence": round(max_div, 6),
        "avg_preflop_divergence": round(avg_preflop_div, 6),
        "avg_flop_divergence": round(avg_flop_div, 6),
        "num_preflop_infosets": len(preflop_divs),
        "num_flop_infosets": len(flop_divs),
        "top_10_divergent": comparison[:10],
        "bottom_10_divergent": comparison[-10:],
    }


def diagnose_exploitation_vs_robustness(eval_results: dict) -> dict:
    """Analyze whether Nash play sacrifices exploitation for robustness."""
    weak_opponents = ["heuristic"]
    strong_opponents = ["cfr", "adaptive_value", "modulated_value", "entropy_ac"]

    weak_scores = [eval_results[name]["avg_chips_per_round"]
                   for name in weak_opponents if name in eval_results]
    strong_scores = [eval_results[name]["avg_chips_per_round"]
                     for name in strong_opponents if name in eval_results]
    all_scores = [v["avg_chips_per_round"] for v in eval_results.values()]

    return {
        "vs_weak_avg": round(sum(weak_scores) / len(weak_scores), 4) if weak_scores else None,
        "vs_strong_avg": round(sum(strong_scores) / len(strong_scores), 4) if strong_scores else None,
        "overall_avg": round(sum(all_scores) / len(all_scores), 4) if all_scores else None,
        "worst_case": round(min(all_scores), 4) if all_scores else None,
        "best_case": round(max(all_scores), 4) if all_scores else None,
        "std": round(float(np.std(all_scores)), 4) if all_scores else None,
    }


# ──────────────────────────────────────────────
# Nash value distribution analysis
# ──────────────────────────────────────────────

def analyze_nash_values(trainer: NashTrainer) -> dict:
    """Analyze the distribution and properties of Nash values."""
    values = list(trainer.nash_values.values())
    keys = list(trainer.nash_values.keys())

    # Basic stats
    val_arr = np.array(values)
    stats = {
        "count": len(values),
        "mean": round(float(val_arr.mean()), 6),
        "std": round(float(val_arr.std()), 6),
        "min": round(float(val_arr.min()), 6),
        "max": round(float(val_arr.max()), 6),
        "median": round(float(np.median(val_arr)), 6),
    }

    # Per-hand breakdown
    hand_values = {"J": [], "Q": [], "K": []}
    for key, val in zip(keys, values):
        hand = key[0]
        if hand in hand_values:
            hand_values[hand].append(val)

    hand_stats = {}
    for hand, vals in hand_values.items():
        if vals:
            arr = np.array(vals)
            hand_stats[hand] = {
                "count": len(vals),
                "mean": round(float(arr.mean()), 6),
                "min": round(float(arr.min()), 6),
                "max": round(float(arr.max()), 6),
            }

    # Pre-flop vs flop
    preflop_vals = [v for k, v in zip(keys, values) if "/" not in k]
    flop_vals = [v for k, v in zip(keys, values) if "/" in k]

    round_stats = {
        "preflop": {
            "count": len(preflop_vals),
            "mean": round(float(np.mean(preflop_vals)), 6) if preflop_vals else None,
        },
        "flop": {
            "count": len(flop_vals),
            "mean": round(float(np.mean(flop_vals)), 6) if flop_vals else None,
        }
    }

    return {
        "overall": stats,
        "per_hand": hand_stats,
        "per_round": round_stats,
    }


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def main():
    print("=" * 70)
    print("  ROUND 4 - DIRECTION 2: NASH VALUE NETWORK")
    print("  Training a neural value net on exact CFR equilibrium values")
    print("=" * 70)

    start_time = time.time()
    results = {
        "experiment": "round4_nash_value_network",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "config": {
            "cfr_iterations": CFR_ITERATIONS,
            "training_epochs": TRAINING_EPOCHS,
            "learning_rate": LEARNING_RATE,
            "eval_rounds_per_position": EVAL_ROUNDS,
            "model_architecture": "15 -> 64 -> 64 -> 1",
        },
    }

    # ── Phase 1: Train Nash Value Network ────────────────────
    print(f"\n{'#'*70}")
    print(f"  PHASE 1: TRAINING NASH VALUE NETWORK")
    print(f"  CFR iterations: {CFR_ITERATIONS}")
    print(f"  Training epochs: {TRAINING_EPOCHS}")
    print(f"  Learning rate: {LEARNING_RATE}")
    print(f"{'#'*70}\n")

    agent = NashValueAgent()
    trainer = NashTrainer(agent, cfr_iterations=CFR_ITERATIONS, learning_rate=LEARNING_RATE)

    training_result = trainer.run_full_pipeline(
        epochs=TRAINING_EPOCHS,
        save_path=MODEL_PATH,
    )
    results["training"] = training_result

    # ── Phase 2: Analyze Nash Values ─────────────────────────
    print(f"\n{'#'*70}")
    print(f"  PHASE 2: NASH VALUE ANALYSIS")
    print(f"{'#'*70}\n")

    value_analysis = analyze_nash_values(trainer)
    results["nash_value_analysis"] = value_analysis

    print(f"  Total infosets: {value_analysis['overall']['count']}")
    print(f"  Value range: [{value_analysis['overall']['min']}, {value_analysis['overall']['max']}]")
    print(f"  Mean value: {value_analysis['overall']['mean']}")
    print(f"  Per hand:")
    for hand, stats in value_analysis["per_hand"].items():
        print(f"    {hand}: mean={stats['mean']:.4f}, range=[{stats['min']:.4f}, {stats['max']:.4f}], n={stats['count']}")

    # ── Phase 3: Diagnose Approximation Quality ──────────────
    print(f"\n{'#'*70}")
    print(f"  PHASE 3: APPROXIMATION QUALITY DIAGNOSIS")
    print(f"{'#'*70}\n")

    approx_diag = diagnose_approximation(trainer)
    results["approximation_diagnosis"] = approx_diag

    print(f"  MSE on all infosets:   {approx_diag['mse']:.8f}")
    print(f"  MAE on all infosets:   {approx_diag['mae']:.6f}")
    print(f"  Max error:             {approx_diag['max_error']:.6f}")
    print(f"\n  Worst-approximated infosets:")
    for entry in approx_diag["worst_10"][:5]:
        print(f"    {entry['key']:30s}  true={entry['true']:+.4f}  pred={entry['pred']:+.4f}  err={entry['error']:.4f}")

    # ── Phase 4: Nash vs Self-Play Comparison ────────────────
    print(f"\n{'#'*70}")
    print(f"  PHASE 4: NASH vs SELF-PLAY VALUE COMPARISON")
    print(f"{'#'*70}\n")

    nash_vs_sp = diagnose_nash_vs_selfplay(trainer)
    results["nash_vs_selfplay"] = nash_vs_sp

    if not nash_vs_sp.get("skipped"):
        print(f"  Avg divergence (Nash vs self-play): {nash_vs_sp['avg_divergence']:.4f}")
        print(f"  Max divergence:                     {nash_vs_sp['max_divergence']:.4f}")
        print(f"  Preflop avg divergence:             {nash_vs_sp['avg_preflop_divergence']:.4f}")
        print(f"  Flop avg divergence:                {nash_vs_sp['avg_flop_divergence']:.4f}")
        print(f"\n  Most divergent infosets (Nash vs self-play):")
        for entry in nash_vs_sp["top_10_divergent"][:5]:
            print(f"    {entry['key']:30s}  nash={entry['nash_true']:+.4f}  sp={entry['other_pred']:+.4f}  div={entry['divergence']:.4f}")
    else:
        print(f"  Skipped: {nash_vs_sp.get('reason', 'unknown')}")

    # ── Phase 5: Tournament Evaluation ───────────────────────
    print(f"\n{'#'*70}")
    print(f"  PHASE 5: TOURNAMENT EVALUATION")
    print(f"  {EVAL_ROUNDS} rounds per position ({EVAL_ROUNDS*2} total per matchup)")
    print(f"{'#'*70}\n")

    nash_agent = NashValueAgent(model_path=MODEL_PATH)
    nash_agent.set_train_mode(False)

    eval_results = {}
    for opp_name, opp_config in OPPONENTS.items():
        try:
            opponent = load_opponent(opp_name, opp_config)
            result = manual_evaluate(nash_agent, opponent, num_rounds=EVAL_ROUNDS)
            eval_results[opp_name] = result
            print(f"  Nash vs {opp_name:20s}: {result['avg_chips_per_round']:+.4f} chips/round")
        except Exception as e:
            print(f"  Nash vs {opp_name:20s}: ERROR - {e}")
            eval_results[opp_name] = {"error": str(e), "avg_chips_per_round": 0.0}

    results["evaluation"] = eval_results

    # Also evaluate using the framework's evaluate_agents for cross-validation
    print(f"\n  Cross-validation using evaluate_agents (PokerSession-based):")
    eval_results_framework = {}
    for opp_name, opp_config in OPPONENTS.items():
        try:
            opponent = load_opponent(opp_name, opp_config)
            result = evaluate_agents(nash_agent, opponent, num_rounds=EVAL_ROUNDS * 2)
            eval_results_framework[opp_name] = {
                "avg_chips_per_round": round(result.agent_0_avg_chips, 4),
                "total_chips": round(result.agent_0_total_chips, 2),
            }
            print(f"  Nash vs {opp_name:20s}: {result.agent_0_avg_chips:+.4f} chips/round (framework)")
        except Exception as e:
            print(f"  Nash vs {opp_name:20s}: ERROR - {e}")

    results["evaluation_framework"] = eval_results_framework

    # ── Phase 6: Exploitation vs Robustness Analysis ─────────
    print(f"\n{'#'*70}")
    print(f"  PHASE 6: EXPLOITATION vs ROBUSTNESS ANALYSIS")
    print(f"{'#'*70}\n")

    expl_rob = diagnose_exploitation_vs_robustness(eval_results)
    results["exploitation_vs_robustness"] = expl_rob

    print(f"  vs Weak opponents avg:   {expl_rob['vs_weak_avg']}")
    print(f"  vs Strong opponents avg: {expl_rob['vs_strong_avg']}")
    print(f"  Overall avg:             {expl_rob['overall_avg']}")
    print(f"  Worst case:              {expl_rob['worst_case']}")
    print(f"  Best case:               {expl_rob['best_case']}")
    print(f"  Std deviation:           {expl_rob['std']}")

    # Robustness metrics
    opponent_scores = {name: r["avg_chips_per_round"] for name, r in eval_results.items()
                       if isinstance(r.get("avg_chips_per_round"), (int, float))}
    robustness = compute_robustness_metrics(opponent_scores)
    results["robustness_metrics"] = robustness

    print(f"\n  Robustness score: {robustness['robustness']}")
    print(f"  (avg={robustness['avg']}, worst={robustness['worst_case']}, std={robustness['std']})")

    # ── Summary ──────────────────────────────────────────────
    elapsed = time.time() - start_time
    results["total_time_s"] = round(elapsed, 1)

    print(f"\n{'='*70}")
    print(f"  EXPERIMENT COMPLETE")
    print(f"  Total time: {elapsed:.1f}s")
    print(f"  Model saved to: {MODEL_PATH}")
    print(f"  Results saved to: {RESULTS_PATH}")
    print(f"{'='*70}\n")

    # Summary table
    print(f"  {'Opponent':>20s}  {'Chips/Round':>12s}")
    print(f"  {'-'*20}  {'-'*12}")
    for name in sorted(eval_results.keys()):
        r = eval_results[name]
        if isinstance(r.get("avg_chips_per_round"), (int, float)):
            print(f"  {name:>20s}  {r['avg_chips_per_round']:+12.4f}")

    # Save results
    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\nResults saved to {RESULTS_PATH}")
    return results


if __name__ == "__main__":
    main()
