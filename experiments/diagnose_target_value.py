"""
Diagnostic experiments for TargetValueAgent underperformance.

Hypotheses:
  H1: Self-play non-stationarity makes frozen targets stale/wrong
  H2: Sync interval (100 grad steps) is too infrequent — target is extremely stale
  H3: Sync interval is too frequent — main network hasn't learned enough
  H4: With gamma=1 and terminal-only rewards, value function changes rapidly
  H5: Implementation bug

Experiments:
  1. Verify target network is actually frozen between syncs
  2. Measure weight divergence (main vs target) at sync points
  3. Sweep sync intervals: 1, 5, 10, 50, 100, 500 (sync=1 = no target network)
  4. Measure value function divergence on probe states between main & target
  5. Compare against parent ValueBasedAgent as control

Training capped at 5K episodes for speed.
"""

import sys
import os
import copy
import json
import torch
import numpy as np
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.agents.target_value import TargetValueAgent
from src.agents.value_based import ValueBasedAgent
from src.training.target_value_trainer import TargetValueTrainer
from src.training.value_based_trainer import SelfPlayTrainer
from src.training.evaluation import quick_evaluate
from src.agents.heuristic import HeuristicAgent
from src.engine.leduc_game import LeducGame, Action
from src.engine.observation import Observation


# ─── Helpers ───────────────────────────────────────────────────────────

def get_param_vector(model):
    """Flatten all model parameters into a single vector."""
    return torch.cat([p.data.flatten() for p in model.parameters()])


def weight_distance(model_a, model_b):
    """L2 distance between two models' parameter vectors."""
    va = get_param_vector(model_a)
    vb = get_param_vector(model_b)
    return (va - vb).norm().item()


def generate_probe_states(agent, n=50):
    """Generate diverse game states for probing value estimates."""
    game = LeducGame()
    probe_states = []
    for _ in range(n * 5):  # oversample, keep unique
        game.reset()
        steps = np.random.randint(0, 6)
        for _ in range(steps):
            if game.is_finished:
                break
            obs = game.get_observation(viewer_id=game.current_player)
            action = np.random.choice(obs.legal_actions)
            game.step(action)
        if not game.is_finished:
            obs = game.get_observation(viewer_id=game.current_player)
            encoded = agent.encode_observation(obs, viewer_id=game.current_player)
            probe_states.append(encoded)
        if len(probe_states) >= n:
            break
    return probe_states


def evaluate_agent(agent, n_games=200):
    """Evaluate agent vs heuristic."""
    agent.set_train_mode(False)
    opponent = HeuristicAgent()
    avg = quick_evaluate(agent, opponent, num_rounds=n_games)
    return avg


# ─── Experiment 1: Verify target freezing ──────────────────────────────

def experiment_1_verify_freezing():
    """
    Verify that the target network is actually frozen between syncs.
    Train for a small number of steps and check target params don't change.
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 1: Verify target network is frozen between syncs")
    print("=" * 70)

    agent = TargetValueAgent()
    trainer = TargetValueTrainer(agent, learning_rate=1e-4, target_sync_every=50)

    # Snapshot target params before training
    target_before = get_param_vector(agent.target_model).clone()

    # Train for 32 episodes (1 gradient step, well before sync at 50)
    agent.set_train_mode(True)
    batch_data = []
    for _ in range(32):
        ep = trainer.collect_episode()
        batch_data.append(ep)
    trainer.update_model(batch_data)

    # Check target hasn't changed
    target_after = get_param_vector(agent.target_model)
    main_after = get_param_vector(agent.model)

    target_change = (target_before - target_after).norm().item()
    main_target_dist = (main_after - target_after).norm().item()

    print(f"  Target param change after 1 grad step: {target_change:.8f}")
    print(f"  Main-Target distance after 1 grad step: {main_target_dist:.6f}")
    print(f"  Target requires_grad: {any(p.requires_grad for p in agent.target_model.parameters())}")

    frozen_ok = target_change < 1e-10
    print(f"\n  RESULT: Target is {'FROZEN (correct)' if frozen_ok else 'NOT FROZEN (BUG!)'}")

    return {
        "target_change": target_change,
        "main_target_distance": main_target_dist,
        "target_frozen": frozen_ok,
    }


# ─── Experiment 2: Weight divergence over training ─────────────────────

def experiment_2_weight_divergence():
    """
    Track how far the main network drifts from the target between syncs.
    This measures "staleness" of the target network.
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 2: Weight divergence (main vs target) over training")
    print("=" * 70)

    agent = TargetValueAgent()
    trainer = TargetValueTrainer(agent, learning_rate=1e-4, target_sync_every=100)
    agent.set_train_mode(True)

    divergence_log = []
    batch_data = []
    episodes_done = 0
    grad_steps = 0
    total_episodes = 3000
    batch_size = 32

    for i in range(total_episodes):
        ep = trainer.collect_episode()
        batch_data.append(ep)
        episodes_done += 1

        if len(batch_data) >= batch_size:
            loss = trainer.update_model(batch_data)
            batch_data = []
            grad_steps += 1

            dist = weight_distance(agent.model, agent.target_model)
            divergence_log.append({
                "grad_step": grad_steps,
                "episode": episodes_done,
                "distance": dist,
                "loss": loss,
                "just_synced": (grad_steps % 100 == 0),
            })

    # Analyze
    sync_dists = [d["distance"] for d in divergence_log if d["just_synced"]]
    presync_dists = [divergence_log[i - 1]["distance"]
                     for i, d in enumerate(divergence_log)
                     if d["just_synced"] and i > 0]
    all_dists = [d["distance"] for d in divergence_log]

    print(f"\n  Total gradient steps: {grad_steps}")
    print(f"  Number of syncs: {len(sync_dists)}")
    print(f"\n  Distance stats (all steps):")
    print(f"    Mean:  {np.mean(all_dists):.4f}")
    print(f"    Max:   {np.max(all_dists):.4f}")
    print(f"    Min:   {np.min(all_dists):.4f}")
    if presync_dists:
        print(f"\n  Distance just BEFORE sync (peak staleness):")
        print(f"    Values: {[f'{d:.4f}' for d in presync_dists]}")
    if sync_dists:
        print(f"\n  Distance just AFTER sync (should be ~0):")
        print(f"    Values: {[f'{d:.4f}' for d in sync_dists]}")

    return {
        "divergence_log": divergence_log,
        "presync_distances": presync_dists,
        "postsync_distances": sync_dists,
        "mean_distance": float(np.mean(all_dists)),
        "max_distance": float(np.max(all_dists)),
    }


# ─── Experiment 3: Sync interval sweep ────────────────────────────────

def experiment_3_sync_interval_sweep():
    """
    Sweep sync intervals from 1 (= no target network) to 500.
    If sync=1 recovers parent performance, staleness is the problem.
    Also run parent ValueBasedAgent as control.
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 3: Sync interval sweep (+ parent control)")
    print("=" * 70)

    total_episodes = 5000
    batch_size = 32
    n_eval = 300

    results = {}

    # Parent ValueBasedAgent (control)
    print("\n  Training parent ValueBasedAgent (control)...")
    parent_agent = ValueBasedAgent()
    parent_trainer = SelfPlayTrainer(parent_agent, learning_rate=1e-4)
    parent_trainer.train(num_episodes=total_episodes, batch_size=batch_size)
    parent_score = evaluate_agent(parent_agent, n_eval)
    results["parent_no_target"] = parent_score
    print(f"  Parent ValueBasedAgent: {parent_score:+.3f} chips/round")

    # Target agent with different sync intervals
    sync_intervals = [1, 5, 10, 25, 50, 100, 500]

    for sync_every in sync_intervals:
        print(f"\n  Training TargetValueAgent (sync_every={sync_every})...")
        agent = TargetValueAgent()
        trainer = TargetValueTrainer(agent, learning_rate=1e-4, target_sync_every=sync_every)
        trainer.train(num_episodes=total_episodes, batch_size=batch_size)
        score = evaluate_agent(agent, n_eval)
        results[f"target_sync_{sync_every}"] = score
        print(f"  TargetValueAgent (sync={sync_every}): {score:+.3f} chips/round")

    # Summary table
    print("\n" + "-" * 50)
    print("  SYNC INTERVAL SWEEP RESULTS")
    print("-" * 50)
    print(f"  {'Config':<30} {'Avg Chips/Round':>15}")
    print(f"  {'-'*30} {'-'*15}")
    print(f"  {'Parent (no target net)':<30} {results['parent_no_target']:>+15.3f}")
    for sync_every in sync_intervals:
        key = f"target_sync_{sync_every}"
        label = f"Target (sync={sync_every})"
        print(f"  {label:<30} {results[key]:>+15.3f}")

    return results


# ─── Experiment 4: Value function divergence on probe states ───────────

def experiment_4_value_divergence_on_probes():
    """
    At each sync point, evaluate the same set of probe states with both
    main and target networks. Measure how much the value estimates disagree.
    This directly shows how "wrong" the target's bootstrap values are.
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 4: Value function divergence on probe states")
    print("=" * 70)

    agent = TargetValueAgent()
    trainer = TargetValueTrainer(agent, learning_rate=1e-4, target_sync_every=50)
    agent.set_train_mode(True)

    # Generate probe states
    probes = generate_probe_states(agent, n=50)
    print(f"  Generated {len(probes)} probe states")

    divergence_at_sync = []
    batch_data = []
    grad_steps = 0
    total_episodes = 3000
    batch_size = 32

    for i in range(total_episodes):
        ep = trainer.collect_episode()
        batch_data.append(ep)

        if len(batch_data) >= batch_size:
            trainer.update_model(batch_data)
            batch_data = []
            grad_steps += 1

            # At sync point (just BEFORE sync happens in the next step),
            # and right AFTER sync, measure value divergence
            # Actually, sync happens inside update_model, so let's check
            # every step to capture both pre-sync and post-sync

            if grad_steps % 50 == 0 or grad_steps % 50 == 49:
                # Compute main vs target values on probes
                main_vals = []
                target_vals = []
                for p in probes:
                    with torch.no_grad():
                        mv = agent.model(p).item()
                        tv = agent.target_model(p).item()
                    main_vals.append(mv)
                    target_vals.append(tv)

                main_vals = np.array(main_vals)
                target_vals = np.array(target_vals)
                diffs = main_vals - target_vals

                entry = {
                    "grad_step": grad_steps,
                    "is_post_sync": (grad_steps % 50 == 0),
                    "mean_abs_diff": float(np.mean(np.abs(diffs))),
                    "max_abs_diff": float(np.max(np.abs(diffs))),
                    "mean_diff": float(np.mean(diffs)),
                    "std_diff": float(np.std(diffs)),
                    "main_mean": float(np.mean(main_vals)),
                    "target_mean": float(np.mean(target_vals)),
                    "correlation": float(np.corrcoef(main_vals, target_vals)[0, 1])
                                  if np.std(main_vals) > 1e-8 and np.std(target_vals) > 1e-8
                                  else 0.0,
                }
                divergence_at_sync.append(entry)

    # Report
    print(f"\n  Total gradient steps: {grad_steps}")

    pre_sync = [e for e in divergence_at_sync if not e["is_post_sync"]]
    post_sync = [e for e in divergence_at_sync if e["is_post_sync"]]

    print(f"\n  Value divergence BEFORE sync (target is maximally stale):")
    if pre_sync:
        for e in pre_sync[:8]:
            print(f"    Step {e['grad_step']:4d}: mean|diff|={e['mean_abs_diff']:.4f}, "
                  f"max|diff|={e['max_abs_diff']:.4f}, corr={e['correlation']:.3f}")

    print(f"\n  Value divergence AFTER sync (target is fresh):")
    if post_sync:
        for e in post_sync[:8]:
            print(f"    Step {e['grad_step']:4d}: mean|diff|={e['mean_abs_diff']:.4f}, "
                  f"max|diff|={e['max_abs_diff']:.4f}, corr={e['correlation']:.3f}")

    return {
        "pre_sync": pre_sync,
        "post_sync": post_sync,
    }


# ─── Experiment 5: Terminal fraction analysis ──────────────────────────

def experiment_5_terminal_fraction():
    """
    Measure what fraction of TD updates use the actual reward (terminal)
    vs. the target network (bootstrap). If most transitions are terminal,
    the target network barely matters — meaning the problem is something else.
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 5: Terminal vs bootstrap transition fraction")
    print("=" * 70)

    agent = TargetValueAgent()
    agent.set_train_mode(True)
    game = LeducGame()

    terminal_count = 0
    bootstrap_count = 0
    chain_lengths = []

    for _ in range(2000):
        game.reset()
        chains = [[], []]
        while not game.is_finished:
            current_player = game.current_player
            obs = game.get_observation(viewer_id=current_player)
            action = agent.select_action(obs)
            if isinstance(action, tuple):
                action = action[0]
            post_obs, _ = LeducGame.simulate_action(obs, action)
            encoded = agent.encode_observation(post_obs, viewer_id=current_player)
            chains[current_player].append(encoded)
            game.step(action)

        for p_idx in [0, 1]:
            chain = chains[p_idx]
            if not chain:
                continue
            chain_lengths.append(len(chain))
            for t in range(len(chain)):
                if t == len(chain) - 1:
                    terminal_count += 1
                else:
                    bootstrap_count += 1

    total = terminal_count + bootstrap_count
    print(f"\n  Total transitions: {total}")
    print(f"  Terminal (use actual reward): {terminal_count} ({100*terminal_count/total:.1f}%)")
    print(f"  Bootstrap (use target net):   {bootstrap_count} ({100*bootstrap_count/total:.1f}%)")
    print(f"\n  Chain length stats:")
    print(f"    Mean: {np.mean(chain_lengths):.2f}")
    print(f"    Median: {np.median(chain_lengths):.1f}")
    print(f"    Min: {np.min(chain_lengths)}, Max: {np.max(chain_lengths)}")
    print(f"    Distribution: {dict(zip(*np.unique(chain_lengths, return_counts=True)))}")

    return {
        "terminal_count": terminal_count,
        "bootstrap_count": bootstrap_count,
        "terminal_fraction": terminal_count / total,
        "bootstrap_fraction": bootstrap_count / total,
        "mean_chain_length": float(np.mean(chain_lengths)),
        "chain_length_dist": dict(zip(
            [int(x) for x in np.unique(chain_lengths)],
            [int(x) for x in np.unique(chain_lengths, return_counts=True)[1]]
        )),
    }


# ─── Experiment 6: Self-play non-stationarity test ────────────────────

def experiment_6_nonstationarity():
    """
    Direct test of H1: Does the value function change rapidly during self-play?
    Track how much the main network's value predictions change on fixed probe
    states over consecutive gradient steps. Compare to the rate of target divergence.
    If main values shift faster than the target can track, staleness is inevitable.
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 6: Self-play non-stationarity — value churn rate")
    print("=" * 70)

    agent = TargetValueAgent()
    trainer = TargetValueTrainer(agent, learning_rate=1e-4, target_sync_every=100)
    agent.set_train_mode(True)

    probes = generate_probe_states(agent, n=50)
    print(f"  Generated {len(probes)} probe states")

    # Track main model values every gradient step
    value_history = []  # list of arrays, one per grad step
    batch_data = []
    grad_steps = 0

    for i in range(3000):
        ep = trainer.collect_episode()
        batch_data.append(ep)

        if len(batch_data) >= 32:
            trainer.update_model(batch_data)
            batch_data = []
            grad_steps += 1

            # Record main model values on probes
            vals = []
            with torch.no_grad():
                for p in probes:
                    vals.append(agent.model(p).item())
            value_history.append(np.array(vals))

    # Compute step-to-step changes
    value_changes = []
    for i in range(1, len(value_history)):
        diff = value_history[i] - value_history[i - 1]
        value_changes.append({
            "step": i + 1,
            "mean_abs_change": float(np.mean(np.abs(diff))),
            "max_abs_change": float(np.max(np.abs(diff))),
            "mean_change": float(np.mean(diff)),
        })

    # Compute cumulative drift over windows of 10, 50, 100 steps
    print(f"\n  Total gradient steps: {grad_steps}")
    print(f"\n  Per-step value change on probe states:")
    mean_per_step = np.mean([vc["mean_abs_change"] for vc in value_changes])
    max_per_step = np.max([vc["max_abs_change"] for vc in value_changes])
    print(f"    Mean |change|/step: {mean_per_step:.5f}")
    print(f"    Max  |change|/step: {max_per_step:.5f}")

    for window in [10, 50, 100]:
        if len(value_history) > window:
            drifts = []
            for i in range(window, len(value_history)):
                drift = np.mean(np.abs(value_history[i] - value_history[i - window]))
                drifts.append(drift)
            print(f"    Cumulative drift over {window} steps: "
                  f"mean={np.mean(drifts):.4f}, max={np.max(drifts):.4f}")

    return {
        "value_changes": value_changes[:20],  # first 20 for brevity
        "mean_per_step_change": mean_per_step,
        "max_per_step_change": max_per_step,
    }


# ─── Main ─────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("DIAGNOSTIC SUITE: Why does TargetValueAgent underperform?")
    print("=" * 70)

    results = {}

    # Exp 1: Is the target actually frozen?
    results["exp1_freezing"] = experiment_1_verify_freezing()

    # Exp 5: Terminal fraction (fast, informative for interpretation)
    results["exp5_terminal_fraction"] = experiment_5_terminal_fraction()

    # Exp 2: Weight divergence
    results["exp2_weight_divergence"] = {
        k: v for k, v in experiment_2_weight_divergence().items()
        if k != "divergence_log"  # too large for JSON
    }

    # Exp 4: Value divergence on probes
    results["exp4_value_divergence"] = experiment_4_value_divergence_on_probes()

    # Exp 6: Non-stationarity
    results["exp6_nonstationarity"] = experiment_6_nonstationarity()

    # Exp 3: The big one — sync interval sweep
    results["exp3_sync_sweep"] = experiment_3_sync_interval_sweep()

    # ─── Final Summary ──────────────────────────────────────────────
    print("\n\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)

    print(f"\n1. Target frozen: {results['exp1_freezing']['target_frozen']}")
    print(f"   (param change = {results['exp1_freezing']['target_change']:.2e})")

    tf = results["exp5_terminal_fraction"]
    print(f"\n2. Terminal vs Bootstrap fraction:")
    print(f"   Terminal: {tf['terminal_fraction']:.1%}, Bootstrap: {tf['bootstrap_fraction']:.1%}")
    print(f"   Mean chain length: {tf['mean_chain_length']:.2f}")

    print(f"\n3. Weight divergence (main vs target):")
    print(f"   Mean distance: {results['exp2_weight_divergence']['mean_distance']:.4f}")
    print(f"   Max distance:  {results['exp2_weight_divergence']['max_distance']:.4f}")

    print(f"\n4. Value divergence on probes (pre-sync staleness):")
    pre = results["exp4_value_divergence"]["pre_sync"]
    if pre:
        avg_mad = np.mean([e["mean_abs_diff"] for e in pre])
        print(f"   Mean |main - target| at peak staleness: {avg_mad:.4f}")

    print(f"\n5. Non-stationarity (value churn):")
    print(f"   Mean per-step value change: {results['exp6_nonstationarity']['mean_per_step_change']:.5f}")

    print(f"\n6. Sync interval sweep results:")
    sweep = results["exp3_sync_sweep"]
    for k, v in sorted(sweep.items()):
        print(f"   {k}: {v:+.3f} chips/round")

    # Save results
    # Convert numpy types for JSON serialization
    def convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    save_path = os.path.join(os.path.dirname(__file__), "diagnose_target_value_results.json")
    with open(save_path, "w") as f:
        json.dump(results, f, indent=2, default=convert)
    print(f"\nResults saved to {save_path}")


if __name__ == "__main__":
    main()
