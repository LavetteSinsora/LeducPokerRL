"""
Multi-seed experiment to reliably compare TargetValueAgent vs parent.

The single-run experiment was too noisy. This runs 5 seeds per config,
with 5K episodes each, to get statistically meaningful comparisons.

Configs tested:
  - Parent ValueBasedAgent (control)
  - TargetValueAgent sync=1 (equivalent to no target net)
  - TargetValueAgent sync=100 (default)
  - TargetValueAgent sync=500

Also measures:
  - Target staleness: how much target value disagrees with main at evaluation time
  - Value churn: how fast the main network's predictions change per grad step
"""

import sys
import os
import torch
import numpy as np
import random
import json
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.agents.target_value import TargetValueAgent
from src.agents.value_based import ValueBasedAgent
from src.training.target_value_trainer import TargetValueTrainer
from src.training.value_based_trainer import SelfPlayTrainer
from src.training.evaluation import quick_evaluate
from src.agents.heuristic import HeuristicAgent
from src.engine.leduc_game import LeducGame


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def generate_probe_states(agent, n=30, seed=42):
    """Generate deterministic probe states."""
    old_state = random.getstate()
    random.seed(seed)
    game = LeducGame()
    probes = []
    for _ in range(n * 10):
        game.reset()
        steps = random.randint(0, 5)
        for _ in range(steps):
            if game.is_finished:
                break
            obs = game.get_observation(viewer_id=game.current_player)
            action = random.choice(obs.legal_actions)
            game.step(action)
        if not game.is_finished:
            obs = game.get_observation(viewer_id=game.current_player)
            encoded = agent.encode_observation(obs, viewer_id=game.current_player)
            probes.append(encoded)
        if len(probes) >= n:
            break
    random.setstate(old_state)
    return probes


def train_and_evaluate(config_name, agent, trainer, total_episodes=5000,
                       batch_size=32, n_eval=500):
    """Train an agent and return final eval score + diagnostics."""
    agent.set_train_mode(True)

    batch_data = []
    grad_steps = 0
    losses = []

    # For staleness tracking (target agents only)
    staleness_log = []
    has_target = hasattr(agent, 'target_model')

    # Generate probes for value tracking
    probes = generate_probe_states(agent)
    value_snapshots = []

    for i in range(total_episodes):
        ep = trainer.collect_episode()
        batch_data.append(ep)

        if len(batch_data) >= batch_size:
            loss = trainer.update_model(batch_data)
            batch_data = []
            grad_steps += 1
            losses.append(loss)

            # Every 10 grad steps, snapshot value predictions
            if grad_steps % 10 == 0:
                vals = []
                with torch.no_grad():
                    for p in probes:
                        vals.append(agent.model(p).item())
                value_snapshots.append(np.array(vals))

                # If target agent, measure staleness
                if has_target:
                    target_vals = []
                    with torch.no_grad():
                        for p in probes:
                            target_vals.append(agent.target_model(p).item())
                    target_vals = np.array(target_vals)
                    main_vals = np.array(vals)
                    staleness_log.append({
                        "step": grad_steps,
                        "mean_abs_diff": float(np.mean(np.abs(main_vals - target_vals))),
                        "max_abs_diff": float(np.max(np.abs(main_vals - target_vals))),
                    })

    # Evaluate
    agent.set_train_mode(False)
    opponent = HeuristicAgent()
    score = quick_evaluate(agent, opponent, num_rounds=n_eval)

    # Compute value churn
    churn_values = []
    for i in range(1, len(value_snapshots)):
        diff = np.mean(np.abs(value_snapshots[i] - value_snapshots[i-1]))
        churn_values.append(diff)

    mean_churn = float(np.mean(churn_values)) if churn_values else 0.0
    mean_staleness = float(np.mean([s["mean_abs_diff"] for s in staleness_log])) if staleness_log else 0.0
    mean_loss = float(np.mean(losses[-20:])) if losses else 0.0  # last 20 batches

    return {
        "score": score,
        "final_loss": mean_loss,
        "grad_steps": grad_steps,
        "mean_value_churn": mean_churn,
        "mean_target_staleness": mean_staleness,
    }


def main():
    print("=" * 70)
    print("MULTI-SEED DIAGNOSTIC: TargetValueAgent vs Parent")
    print("=" * 70)

    configs = {
        "parent_no_target": {"sync": None},
        "target_sync_1": {"sync": 1},
        "target_sync_100": {"sync": 100},
        "target_sync_500": {"sync": 500},
    }

    seeds = [0, 1, 2, 3, 4]
    total_episodes = 5000
    batch_size = 32
    n_eval = 500

    all_results = {}

    for config_name, config in configs.items():
        print(f"\n{'='*60}")
        print(f"Config: {config_name}")
        print(f"{'='*60}")

        seed_results = []
        for seed in seeds:
            set_seed(seed)
            print(f"  Seed {seed}...", end=" ", flush=True)

            if config["sync"] is None:
                # Parent ValueBasedAgent
                agent = ValueBasedAgent()
                trainer = SelfPlayTrainer(agent, learning_rate=1e-4)
            else:
                agent = TargetValueAgent()
                trainer = TargetValueTrainer(agent, learning_rate=1e-4,
                                             target_sync_every=config["sync"])

            result = train_and_evaluate(config_name, agent, trainer,
                                        total_episodes=total_episodes,
                                        batch_size=batch_size,
                                        n_eval=n_eval)
            seed_results.append(result)
            print(f"score={result['score']:+.3f}, loss={result['final_loss']:.2f}, "
                  f"churn={result['mean_value_churn']:.4f}, "
                  f"staleness={result['mean_target_staleness']:.4f}")

        scores = [r["score"] for r in seed_results]
        churns = [r["mean_value_churn"] for r in seed_results]
        stalenesses = [r["mean_target_staleness"] for r in seed_results]

        all_results[config_name] = {
            "scores": scores,
            "mean_score": float(np.mean(scores)),
            "std_score": float(np.std(scores)),
            "min_score": float(np.min(scores)),
            "max_score": float(np.max(scores)),
            "mean_churn": float(np.mean(churns)),
            "mean_staleness": float(np.mean(stalenesses)),
            "seed_results": seed_results,
        }

    # ─── Summary Table ──────────────────────────────────────────────
    print("\n\n" + "=" * 75)
    print("MULTI-SEED RESULTS SUMMARY (5 seeds x 5K episodes each)")
    print("=" * 75)
    print(f"  {'Config':<25} {'Mean':>8} {'Std':>8} {'Min':>8} {'Max':>8}  {'Churn':>8} {'Staleness':>10}")
    print(f"  {'-'*25} {'-'*8} {'-'*8} {'-'*8} {'-'*8}  {'-'*8} {'-'*10}")

    for config_name in configs:
        r = all_results[config_name]
        print(f"  {config_name:<25} {r['mean_score']:>+8.3f} {r['std_score']:>8.3f} "
              f"{r['min_score']:>+8.3f} {r['max_score']:>+8.3f}  "
              f"{r['mean_churn']:>8.4f} {r['mean_staleness']:>10.4f}")

    # Statistical comparison
    print(f"\n  Key comparisons:")
    parent_scores = all_results["parent_no_target"]["scores"]
    for name in ["target_sync_1", "target_sync_100", "target_sync_500"]:
        target_scores = all_results[name]["scores"]
        diff_mean = np.mean(target_scores) - np.mean(parent_scores)
        # Simple t-test approximation
        pooled_std = np.sqrt((np.var(parent_scores) + np.var(target_scores)) / 2)
        t_stat = diff_mean / (pooled_std * np.sqrt(2/5)) if pooled_std > 0 else 0
        print(f"  {name} vs parent: diff = {diff_mean:+.3f}, t = {t_stat:.2f}")

    # ─── Hypothesis verdicts ─────────────────────────────────────────
    print(f"\n\n{'='*75}")
    print("HYPOTHESIS ANALYSIS")
    print("=" * 75)

    # H5: Bug check
    print(f"\nH5 (Implementation bug): REJECTED")
    print(f"  Target network is correctly frozen (verified in Exp 1).")
    print(f"  Target correctly used for bootstrap, main for prediction.")

    # H1/H4: Self-play non-stationarity
    parent_mean = all_results["parent_no_target"]["mean_score"]
    sync1_mean = all_results["target_sync_1"]["mean_score"]
    sync100_mean = all_results["target_sync_100"]["mean_score"]
    sync500_mean = all_results["target_sync_500"]["mean_score"]

    print(f"\nH1 (Self-play non-stationarity makes targets stale):")
    print(f"  Parent (no target): {parent_mean:+.3f}")
    print(f"  sync=1 (=parent):   {sync1_mean:+.3f}")
    print(f"  sync=100:           {sync100_mean:+.3f}")
    print(f"  sync=500:           {sync500_mean:+.3f}")

    staleness_100 = all_results["target_sync_100"]["mean_staleness"]
    staleness_500 = all_results["target_sync_500"]["mean_staleness"]
    print(f"  Target staleness (mean |V_main - V_target|):")
    print(f"    sync=100: {staleness_100:.4f}")
    print(f"    sync=500: {staleness_500:.4f}")

    # Check if all configs are within noise of each other
    all_means = [all_results[k]["mean_score"] for k in configs]
    all_stds = [all_results[k]["std_score"] for k in configs]
    print(f"\n  Score range: {min(all_means):+.3f} to {max(all_means):+.3f}")
    print(f"  Typical std:  {np.mean(all_stds):.3f}")
    print(f"  Range / std:  {(max(all_means) - min(all_means)) / np.mean(all_stds):.2f}")

    if (max(all_means) - min(all_means)) < 2 * np.mean(all_stds):
        print(f"\n  CONCLUSION: All configs are within noise of each other at 5K episodes.")
        print(f"  The target network neither helps nor hurts significantly at this scale.")
        print(f"  The original tournament difference (-0.31 vs +0.98 at 20K) likely comes from:")
        print(f"    1. High variance in self-play training (all configs vary widely)")
        print(f"    2. The target network adds unnecessary complexity without benefit")
        print(f"    3. In a game with short episodes (chains ~2 steps), most TD updates")
        print(f"       use terminal rewards anyway (46.6% terminal), reducing the")
        print(f"       target network's influence to roughly half of transitions")
    else:
        print(f"\n  CONCLUSION: There is a statistically meaningful difference between configs.")

    # Save
    save_path = os.path.join(os.path.dirname(__file__), "diagnose_target_value_multiseed_results.json")
    def convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj
    with open(save_path, "w") as f:
        json.dump(all_results, f, indent=2, default=convert)
    print(f"\nResults saved to {save_path}")


if __name__ == "__main__":
    main()
