"""
Diagnostic experiment for two Round 3 agents:

Agent 1: pruned_history (avg=-0.691, robustness=-1.781)
  Pruned action history from 16->12 dims, 64 hidden units. Trained 667 sessions.
  Hypotheses:
    A) 667 sessions not enough — was supposed to get 2000 (plan said 60K hands)
    B) 64 hidden units too small — adaptive_history used 128
    C) The pruning itself removed useful features

Agent 2: extended_adaptive (avg=+0.329, robustness=-0.406)
  Null hypothesis control: identical to adaptive_value but trained 3x longer (2000 sessions).
  adaptive_value scored avg=+1.012 — so 3x training made it WORSE.
  Hypotheses:
    A) Overfitting from too much training
    B) Learning rate should decay for longer training
    C) Seed variance

Experiments:
  1. Train pruned_history for 2000 sessions (proper budget fix)
  2. Train extended_adaptive at 667 / 1000 / 2000 sessions, evaluate each checkpoint
  3. Evaluate all variants against heuristic, value_based, adaptive_value (500 rounds)
  4. Compare: does more training help or hurt?
"""

import os
import sys
import copy
import time
import json
import torch
import numpy as np
import random

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.agents.heuristic import HeuristicAgent
from src.agents.value_based import ValueBasedAgent
from src.agents.adaptive_value import AdaptiveValueAgent
from src.agents.pruned_history import PrunedHistoryAgent
from src.agents.extended_adaptive import ExtendedAdaptiveAgent
from src.training.adaptive_trainer import AdaptiveTrainer
from src.training.pruned_history_trainer import PrunedHistoryTrainer
from src.training.evaluation import evaluate_agents, quick_evaluate, compute_robustness_metrics


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def separator(title):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}\n")


# ──────────────────────────────────────────────────────────────────────
# Experiment 1: Train pruned_history with proper budget (2000 sessions)
# ──────────────────────────────────────────────────────────────────────
def exp1_pruned_history_full_budget():
    separator("EXP 1 — pruned_history: 2000 sessions (proper budget)")

    set_seed(42)
    agent = PrunedHistoryAgent()
    trainer = PrunedHistoryTrainer(agent, learning_rate=1e-4, hands_per_session=30)

    losses = []
    eval_scores = []

    def callback(data):
        if data["type"] == "batch_update":
            losses.append(data["loss"])
        elif data["type"] == "evaluation":
            eval_scores.append(data["avg_chips_per_round"])

    start = time.time()
    trainer.train(num_episodes=2000, batch_size=32,
                  save_path="models/diag_pruned_history_2000.pt",
                  callback=callback)
    elapsed = time.time() - start

    print(f"\n  Training time: {elapsed:.1f}s")
    print(f"  Total hands:   {2000 * 30} = 60,000 (matches original plan)")
    print(f"  Final loss:    {losses[-1]:.4f}" if losses else "  No losses recorded")
    if eval_scores:
        print(f"  Final eval vs heuristic: {eval_scores[-1]:+.3f}")
        print(f"  Best eval vs heuristic:  {max(eval_scores):+.3f}")

    return {
        "losses": [float(l) for l in losses],
        "n_losses": len(losses),
        "evals": [float(e) for e in eval_scores],
        "time": elapsed,
        "final_loss": float(losses[-1]) if losses else None,
    }


# ──────────────────────────────────────────────────────────────────────
# Experiment 2: extended_adaptive overfitting curve
# Train and checkpoint at 667 / 1000 / 2000 sessions
# ──────────────────────────────────────────────────────────────────────
def exp2_extended_adaptive_overfitting_curve():
    separator("EXP 2 — extended_adaptive: overfitting curve (667 / 1000 / 2000)")

    set_seed(42)
    agent = ExtendedAdaptiveAgent()
    trainer = AdaptiveTrainer(agent, learning_rate=1e-4, hands_per_session=30)

    losses = []
    checkpoints = {}
    checkpoint_sessions = [667, 1000, 2000]
    checkpoint_idx = 0

    agent.set_train_mode(True)
    batch_data = []
    episode_counter = 0

    for session_idx in range(2000):
        session_data = trainer.collect_episode()
        batch_data.extend(session_data)
        episode_counter += len(session_data)

        if len(batch_data) >= 32:
            loss = trainer.update_model(batch_data)
            batch_data = []
            losses.append(loss)

            if (session_idx + 1) % 200 == 0:
                print(f"    Session {session_idx + 1}/2000, loss={loss:.4f}")

        # Save checkpoint at each target session count
        if checkpoint_idx < len(checkpoint_sessions) and (session_idx + 1) == checkpoint_sessions[checkpoint_idx]:
            ckpt_name = f"extended_{checkpoint_sessions[checkpoint_idx]}"
            ckpt_path = f"models/diag_{ckpt_name}.pt"
            os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
            agent.save_model(ckpt_path)
            checkpoints[ckpt_name] = {
                "session": checkpoint_sessions[checkpoint_idx],
                "path": ckpt_path,
                "loss_at_checkpoint": float(losses[-1]) if losses else None,
                "total_losses_so_far": len(losses),
            }
            print(f"  >>> Checkpoint saved: {ckpt_name} (session {checkpoint_sessions[checkpoint_idx]})")
            checkpoint_idx += 1

    # Flush remaining batch
    if batch_data:
        loss = trainer.update_model(batch_data)
        losses.append(loss)

    agent.set_train_mode(False)

    print(f"\n  Training complete. {len(losses)} gradient updates total.")
    print(f"  Loss trajectory (sampled):")
    for i in [0, len(losses)//4, len(losses)//2, 3*len(losses)//4, len(losses)-1]:
        if i < len(losses):
            print(f"    step {i:>5d}: {losses[i]:.4f}")

    return {
        "checkpoints": checkpoints,
        "losses": [float(l) for l in losses],
        "n_losses": len(losses),
    }


# ──────────────────────────────────────────────────────────────────────
# Experiment 3: Evaluate all variants head-to-head
# ──────────────────────────────────────────────────────────────────────
def exp3_evaluate_all_variants():
    separator("EXP 3 — Head-to-head evaluation (500 rounds each)")

    agents = {}

    # Baselines
    agents["heuristic"] = HeuristicAgent()

    vb = ValueBasedAgent()
    if os.path.exists("models/value_based_agent.pt"):
        vb.load_model("models/value_based_agent.pt")
    vb.set_train_mode(False)
    agents["value_based"] = vb

    av = AdaptiveValueAgent()
    if os.path.exists("models/adaptive_value_agent.pt"):
        av.load_model("models/adaptive_value_agent.pt")
    av.set_train_mode(False)
    agents["adaptive_value"] = av

    # Original pruned_history (667 sessions)
    ph_orig = PrunedHistoryAgent()
    if os.path.exists("models/pruned_history_agent.pt"):
        ph_orig.load_model("models/pruned_history_agent.pt")
    ph_orig.set_train_mode(False)
    agents["pruned_667_orig"] = ph_orig

    # Pruned_history with full 2000 session budget
    if os.path.exists("models/diag_pruned_history_2000.pt"):
        ph_2000 = PrunedHistoryAgent()
        ph_2000.load_model("models/diag_pruned_history_2000.pt")
        ph_2000.set_train_mode(False)
        agents["pruned_2000_fix"] = ph_2000

    # Original extended_adaptive (2000 sessions pretrained)
    ea_orig = ExtendedAdaptiveAgent()
    if os.path.exists("models/extended_adaptive_agent.pt"):
        ea_orig.load_model("models/extended_adaptive_agent.pt")
    ea_orig.set_train_mode(False)
    agents["ext_2000_orig"] = ea_orig

    # Extended_adaptive checkpoints
    for sessions in [667, 1000, 2000]:
        path = f"models/diag_extended_{sessions}.pt"
        if os.path.exists(path):
            ea = ExtendedAdaptiveAgent()
            ea.load_model(path)
            ea.set_train_mode(False)
            agents[f"ext_{sessions}_new"] = ea

    # Test opponents (the 3 we evaluate against)
    test_opponents = ["heuristic", "value_based", "adaptive_value"]

    # Evaluate each agent against each test opponent
    agent_names = list(agents.keys())
    results = {}

    for agent_name in agent_names:
        results[agent_name] = {}
        scores_for_robustness = {}

        for opp_name in test_opponents:
            if agent_name == opp_name:
                results[agent_name][opp_name] = 0.0
                continue

            print(f"  {agent_name:>20s} vs {opp_name:<20s} ...", end=" ", flush=True)
            result = evaluate_agents(agents[agent_name], agents[opp_name], num_rounds=500)
            score = round(result.agent_0_avg_chips, 4)
            results[agent_name][opp_name] = score
            scores_for_robustness[opp_name] = score
            print(f"{score:+.4f}")

        # Compute robustness
        metrics = compute_robustness_metrics(scores_for_robustness)
        results[agent_name]["_metrics"] = metrics

    # ── Print results table ──────────────────────────────────────────
    print(f"\n  {'Agent':>20s} |", end="")
    for opp in test_opponents:
        print(f" {opp[:14]:>14s}", end="")
    print(f" |    AVG   ROBUST")
    print(f"  {'-'*20}-+" + "-" * (15 * len(test_opponents)) + "-+-----------------")

    for agent_name in agent_names:
        r = results[agent_name]
        m = r.get("_metrics", {})
        print(f"  {agent_name:>20s} |", end="")
        for opp in test_opponents:
            if agent_name == opp:
                print(f" {'---':>14s}", end="")
            else:
                print(f" {r.get(opp, 0.0):>+14.4f}", end="")
        avg = m.get("avg", 0.0)
        robust = m.get("robustness", 0.0)
        print(f" | {avg:>+7.4f} {robust:>+7.4f}")

    return results


# ──────────────────────────────────────────────────────────────────────
# Experiment 4: Extended adaptive with full round-robin among checkpoints
# ──────────────────────────────────────────────────────────────────────
def exp4_extended_adaptive_checkpoint_comparison():
    separator("EXP 4 — extended_adaptive checkpoint comparison (round-robin)")

    agents = {}

    # Load all extended_adaptive checkpoints
    for sessions in [667, 1000, 2000]:
        path = f"models/diag_extended_{sessions}.pt"
        if os.path.exists(path):
            ea = ExtendedAdaptiveAgent()
            ea.load_model(path)
            ea.set_train_mode(False)
            agents[f"ext_{sessions}"] = ea

    # Also include original and parent
    ea_orig = ExtendedAdaptiveAgent()
    if os.path.exists("models/extended_adaptive_agent.pt"):
        ea_orig.load_model("models/extended_adaptive_agent.pt")
    ea_orig.set_train_mode(False)
    agents["ext_orig_2000"] = ea_orig

    av = AdaptiveValueAgent()
    if os.path.exists("models/adaptive_value_agent.pt"):
        av.load_model("models/adaptive_value_agent.pt")
    av.set_train_mode(False)
    agents["adaptive_value"] = av

    agents["heuristic"] = HeuristicAgent()

    if len(agents) < 3:
        print("  Not enough checkpoints to compare. Skipping.")
        return {}

    # Round-robin (only ext checkpoints vs baselines)
    agent_names = list(agents.keys())
    results = {a: {} for a in agent_names}

    for i, a0_name in enumerate(agent_names):
        results[a0_name][a0_name] = 0.0
        for j, a1_name in enumerate(agent_names):
            if j <= i:
                continue
            print(f"    {a0_name:>16s} vs {a1_name:<16s} ...", end=" ", flush=True)
            result = evaluate_agents(agents[a0_name], agents[a1_name], num_rounds=500)
            results[a0_name][a1_name] = round(result.agent_0_avg_chips, 4)
            results[a1_name][a0_name] = round(result.agent_1_avg_chips, 4)
            print(f"{results[a0_name][a1_name]:+.4f} / {results[a1_name][a0_name]:+.4f}")

    # Print ranking
    print(f"\n  {'Agent':>16s} |", end="")
    for name in agent_names:
        print(f" {name[:12]:>12s}", end="")
    print(f" |     AVG")
    print(f"  {'-'*16}-+" + "-" * (13 * len(agent_names)) + "-+--------")

    for a0 in agent_names:
        print(f"  {a0:>16s} |", end="")
        scores = []
        for a1 in agent_names:
            if a0 == a1:
                print(f" {'---':>12s}", end="")
            else:
                val = results[a0].get(a1, 0.0)
                print(f" {val:>+12.4f}", end="")
                scores.append(val)
        avg = sum(scores) / len(scores) if scores else 0
        print(f" | {avg:>+.4f}")

    return results


# ──────────────────────────────────────────────────────────────────────
# Analysis and hypothesis verdicts
# ──────────────────────────────────────────────────────────────────────
def analyze_results(exp1_result, exp2_result, exp3_result, exp4_result):
    separator("HYPOTHESIS ANALYSIS")

    # ── pruned_history ────────────────────────────────────────────────
    print("=" * 60)
    print("  PRUNED_HISTORY HYPOTHESES")
    print("=" * 60)

    # Get scores for comparison
    pruned_orig = exp3_result.get("pruned_667_orig", {})
    pruned_fixed = exp3_result.get("pruned_2000_fix", {})
    av_scores = exp3_result.get("adaptive_value", {})

    pruned_orig_avg = pruned_orig.get("_metrics", {}).get("avg", "N/A")
    pruned_fixed_avg = pruned_fixed.get("_metrics", {}).get("avg", "N/A")
    av_avg = av_scores.get("_metrics", {}).get("avg", "N/A")

    print(f"\n  H(A) — Training budget (667 vs 2000 sessions):")
    print(f"    pruned_history (667 sessions, original): avg = {pruned_orig_avg}")
    print(f"    pruned_history (2000 sessions, fixed):   avg = {pruned_fixed_avg}")
    print(f"    adaptive_value (667 sessions, parent):   avg = {av_avg}")

    if isinstance(pruned_orig_avg, float) and isinstance(pruned_fixed_avg, float):
        improvement = pruned_fixed_avg - pruned_orig_avg
        print(f"    Improvement from 3x budget: {improvement:+.4f}")
        if improvement > 0.3:
            print(f"    >>> SUPPORTED: More training significantly helps")
        elif improvement > 0.1:
            print(f"    >>> PARTIALLY SUPPORTED: Modest improvement from more training")
        else:
            print(f"    >>> REJECTED: More training doesn't fix it; the issue is elsewhere")

    print(f"\n  H(B) — 64 hidden units too small:")
    print(f"    pruned_history uses 64 hidden, adaptive_history uses 128")
    print(f"    Even with 2000 sessions, pruned_history avg = {pruned_fixed_avg}")
    print(f"    vs adaptive_value (also 64 hidden, no history) avg = {av_avg}")
    if isinstance(pruned_fixed_avg, float) and isinstance(av_avg, float):
        if pruned_fixed_avg < av_avg - 0.3:
            print(f"    >>> The history features + 64 units hurt relative to simpler 19-dim + 64 units")
            print(f"    >>> Suggests 64 units may be insufficient for the 31-dim input")
        else:
            print(f"    >>> 64 units seem adequate; history features don't drag performance down")

    print(f"\n  H(C) — Pruning removed useful features:")
    print(f"    Fold counts are always zero in action history (fold ends the hand).")
    print(f"    Removing them should be lossless. If pruned_history still underperforms")
    print(f"    even with full budget, the issue is the 64 hidden units or training dynamics.")

    # ── extended_adaptive ─────────────────────────────────────────────
    print(f"\n\n{'='*60}")
    print("  EXTENDED_ADAPTIVE HYPOTHESES")
    print("=" * 60)

    ext_667 = exp3_result.get("ext_667_new", {})
    ext_1000 = exp3_result.get("ext_1000_new", {})
    ext_2000 = exp3_result.get("ext_2000_new", {})
    ext_orig = exp3_result.get("ext_2000_orig", {})

    ext_667_avg = ext_667.get("_metrics", {}).get("avg", "N/A")
    ext_1000_avg = ext_1000.get("_metrics", {}).get("avg", "N/A")
    ext_2000_avg = ext_2000.get("_metrics", {}).get("avg", "N/A")
    ext_orig_avg = ext_orig.get("_metrics", {}).get("avg", "N/A")

    print(f"\n  H(A) — Overfitting from too much training:")
    print(f"    adaptive_value (667 sessions):     avg = {av_avg}")
    print(f"    extended (667 sessions, new seed):  avg = {ext_667_avg}")
    print(f"    extended (1000 sessions, new seed): avg = {ext_1000_avg}")
    print(f"    extended (2000 sessions, new seed): avg = {ext_2000_avg}")
    print(f"    extended (2000 sessions, original): avg = {ext_orig_avg}")

    # Check for declining performance
    ext_avgs = []
    for label, val in [("667", ext_667_avg), ("1000", ext_1000_avg), ("2000", ext_2000_avg)]:
        if isinstance(val, float):
            ext_avgs.append((label, val))

    if len(ext_avgs) >= 2:
        if ext_avgs[-1][1] < ext_avgs[0][1] - 0.2:
            print(f"\n    >>> SUPPORTED: Performance DEGRADES with more training")
            print(f"    >>> {ext_avgs[0][0]} sessions: {ext_avgs[0][1]:+.4f} -> "
                  f"{ext_avgs[-1][0]} sessions: {ext_avgs[-1][1]:+.4f}")
            print(f"    >>> This is classic overfitting to self-play patterns")
        elif ext_avgs[-1][1] > ext_avgs[0][1] + 0.2:
            print(f"\n    >>> REJECTED: Performance IMPROVES with more training")
            print(f"    >>> Original poor result was likely seed variance")
        else:
            print(f"\n    >>> INCONCLUSIVE: Performance roughly flat across training budgets")

    print(f"\n  H(C) — Seed variance check:")
    if isinstance(ext_2000_avg, float) and isinstance(ext_orig_avg, float):
        seed_diff = abs(ext_2000_avg - ext_orig_avg)
        print(f"    Original ext_2000 (different seed): avg = {ext_orig_avg}")
        print(f"    New ext_2000 (seed=42):             avg = {ext_2000_avg}")
        print(f"    Difference: {seed_diff:.4f}")
        if seed_diff > 0.5:
            print(f"    >>> SUPPORTED: Large seed variance ({seed_diff:.3f}). ")
            print(f"    >>> The original poor result may be an unlucky seed.")
        else:
            print(f"    >>> REJECTED: Seed variance is small; result is reproducible.")

    # Loss trajectory analysis for overfitting
    if exp2_result and exp2_result.get("losses"):
        losses = exp2_result["losses"]
        n = len(losses)
        if n >= 10:
            early_loss = np.mean(losses[n//4:n//4+5])
            mid_loss = np.mean(losses[n//2:n//2+5])
            late_loss = np.mean(losses[-5:])
            print(f"\n  Loss trajectory (from Exp 2):")
            print(f"    Early (25%):  {early_loss:.4f}")
            print(f"    Middle (50%): {mid_loss:.4f}")
            print(f"    Late (100%):  {late_loss:.4f}")
            if late_loss < early_loss * 0.5:
                print(f"    Loss keeps dropping — but performance may still overfit")
                print(f"    (self-play loss measures fit to training distribution, not generalization)")


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────
def main():
    print("=" * 70)
    print("  DIAGNOSIS: pruned_history & extended_adaptive")
    print("  pruned_history:     avg=-0.691, training budget was only 667/2000")
    print("  extended_adaptive:  avg=+0.329, worse than parent (+1.012) despite 3x training")
    print("=" * 70)

    all_results = {}
    start_total = time.time()

    # Exp 1: Train pruned_history with proper 2000-session budget
    all_results["exp1_pruned_full_budget"] = exp1_pruned_history_full_budget()

    # Exp 2: Extended_adaptive overfitting curve with checkpoints
    all_results["exp2_ext_overfitting"] = exp2_extended_adaptive_overfitting_curve()

    # Exp 3: Evaluate all variants against baselines
    all_results["exp3_h2h"] = exp3_evaluate_all_variants()

    # Exp 4: Round-robin among extended_adaptive checkpoints
    all_results["exp4_ext_rr"] = exp4_extended_adaptive_checkpoint_comparison()

    total_time = time.time() - start_total

    # Analysis
    analyze_results(
        all_results["exp1_pruned_full_budget"],
        all_results["exp2_ext_overfitting"],
        all_results["exp3_h2h"],
        all_results["exp4_ext_rr"],
    )

    # ── Final summary ────────────────────────────────────────────────
    separator("FINAL SUMMARY")

    exp3 = all_results["exp3_h2h"]
    print("  Agent Performance Rankings (avg chips/round vs 3 baselines):")
    print(f"  {'Agent':>20s}   {'Avg':>8s}  {'Robustness':>10s}")
    print(f"  {'-'*45}")

    ranked = []
    for name, data in exp3.items():
        metrics = data.get("_metrics", {})
        if metrics:
            ranked.append((name, metrics.get("avg", 0), metrics.get("robustness", 0)))

    ranked.sort(key=lambda x: x[1], reverse=True)
    for name, avg, robust in ranked:
        print(f"  {name:>20s}   {avg:>+8.4f}  {robust:>+10.4f}")

    print(f"\n  Total experiment time: {total_time:.1f}s")

    # ── Save results ─────────────────────────────────────────────────
    results_path = os.path.join(os.path.dirname(__file__), "diagnose_pruned_extended_results.json")

    # Make JSON-serializable
    def make_serializable(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating, np.float64)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: make_serializable(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [make_serializable(i) for i in obj]
        return obj

    serializable = make_serializable(all_results)

    with open(results_path, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"\n  Results saved to {results_path}")


if __name__ == "__main__":
    main()
