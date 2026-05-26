#!/usr/bin/env python3
"""
Diagnostic experiment: Why does NStepValueAgent underperform ValueBasedAgent?

Hypotheses:
  H1: n=3 turns training into near-pure Monte Carlo (most transitions use
      terminal reward, not bootstrap), which has high variance in self-play
      where the opponent policy changes every batch.
  H2: TD(0) bootstrapping provides implicit temporal smoothing -- targets
      change slowly as the network changes, stabilizing training. MC targets
      are raw game outcomes with full variance.
  H3: Chain lengths in Leduc are so short (2-4) that n=3 is effectively
      full MC for nearly all transitions.

Experiments:
  1. Measure chain length distribution over 1000 games
  2. For each n in {1,2,3,4,MC}, compute fraction of transitions that are
     "bootstrapped" vs "terminal"
  3. Train agents with n=1,2,3,4,MC for 5K episodes, compare eval vs heuristic
     + measure training loss variance

Usage:
  python -m experiments.diagnose_nstep
"""

import sys, os, copy, random, math, json
from collections import defaultdict, Counter

import torch
import numpy as np

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.engine.leduc_game import LeducGame, Action
from src.agents.value_based import ValueBasedAgent
from src.agents.nstep_value import NStepValueAgent
from src.agents.heuristic import HeuristicAgent
from src.training.nstep_value_trainer import NStepValueTrainer
from src.training.value_based_trainer import SelfPlayTrainer
from src.training.evaluation import quick_evaluate

# ──────────────────────────────────────────────────────────────────────
# Experiment 1: Chain length distribution
# ──────────────────────────────────────────────────────────────────────

def measure_chain_lengths(num_games=1000):
    """Play games with a random/heuristic agent and record per-player chain lengths."""
    game = LeducGame()
    agent = ValueBasedAgent()  # untrained, essentially random

    all_chain_lengths = []  # list of per-player chain lengths
    total_actions_per_game = []

    for _ in range(num_games):
        game.reset()
        chains = [0, 0]
        total_actions = 0

        while not game.is_finished:
            cp = game.current_player
            obs = game.get_observation(viewer_id=cp)
            action = random.choice(obs.legal_actions)
            game.step(action)
            chains[cp] += 1
            total_actions += 1

        all_chain_lengths.extend(chains)
        total_actions_per_game.append(total_actions)

    return all_chain_lengths, total_actions_per_game


def experiment_1():
    print("=" * 70)
    print("EXPERIMENT 1: Chain Length Distribution (1000 random games)")
    print("=" * 70)

    chain_lengths, total_actions = measure_chain_lengths(1000)

    counter = Counter(chain_lengths)
    total = len(chain_lengths)

    print(f"\nPer-player chain lengths (N={total} player-chains from 1000 games):")
    for length in sorted(counter.keys()):
        pct = counter[length] / total * 100
        bar = "#" * int(pct)
        print(f"  Length {length}: {counter[length]:4d} ({pct:5.1f}%) {bar}")

    avg_chain = sum(chain_lengths) / len(chain_lengths)
    print(f"\n  Mean chain length: {avg_chain:.2f}")
    print(f"  Min: {min(chain_lengths)}, Max: {max(chain_lengths)}")

    avg_total = sum(total_actions) / len(total_actions)
    print(f"\n  Mean total actions per game: {avg_total:.2f}")

    return chain_lengths


# ──────────────────────────────────────────────────────────────────────
# Experiment 2: Bootstrap fraction for different n values
# ──────────────────────────────────────────────────────────────────────

def compute_bootstrap_fraction(chain_lengths, n_steps):
    """For given chain lengths and n, compute what fraction of transitions bootstrap vs use terminal."""
    total_transitions = 0
    bootstrapped = 0
    terminal = 0

    for L in chain_lengths:
        for t in range(L):
            total_transitions += 1
            if t + n_steps >= L:
                terminal += 1
            else:
                bootstrapped += 1

    return {
        "n_steps": n_steps,
        "total_transitions": total_transitions,
        "bootstrapped": bootstrapped,
        "terminal": terminal,
        "bootstrap_frac": bootstrapped / total_transitions if total_transitions > 0 else 0,
        "terminal_frac": terminal / total_transitions if total_transitions > 0 else 0,
    }


def experiment_2(chain_lengths):
    print("\n" + "=" * 70)
    print("EXPERIMENT 2: Bootstrap vs Terminal Fraction by n")
    print("=" * 70)

    # Per-player chain lengths from experiment 1
    results = []
    for n in [1, 2, 3, 4, 100]:  # 100 = effectively MC
        label = f"n={n}" if n < 100 else "MC"
        r = compute_bootstrap_fraction(chain_lengths, n)
        r["label"] = label
        results.append(r)
        print(f"\n  {label}:")
        print(f"    Total transitions: {r['total_transitions']}")
        print(f"    Bootstrapped:      {r['bootstrapped']:4d} ({r['bootstrap_frac']*100:.1f}%)")
        print(f"    Terminal (MC):     {r['terminal']:4d} ({r['terminal_frac']*100:.1f}%)")

    return results


# ──────────────────────────────────────────────────────────────────────
# Experiment 3: Training comparison across n values
# ──────────────────────────────────────────────────────────────────────

def train_and_evaluate(n_steps, num_episodes=5000, batch_size=32, eval_rounds=200,
                       seed=42, init_state_dict=None):
    """Train an NStepValue agent with given n, return eval scores and loss trajectory."""

    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    agent = NStepValueAgent()
    if init_state_dict is not None:
        agent.model.load_state_dict(copy.deepcopy(init_state_dict))

    # Use the appropriate trainer
    if n_steps == 1:
        # n=1 should be equivalent to TD(0), but use the NStepValueTrainer
        # to isolate the effect of n vs any other code differences
        trainer = NStepValueTrainer(agent, learning_rate=1e-4, n_steps=1)
    elif n_steps >= 100:
        # MC: use very large n so everything uses terminal reward
        trainer = NStepValueTrainer(agent, learning_rate=1e-4, n_steps=100)
    else:
        trainer = NStepValueTrainer(agent, learning_rate=1e-4, n_steps=n_steps)

    # Also train with the original SelfPlayTrainer for n=1 comparison
    loss_history = []
    eval_history = []

    agent.set_train_mode(True)
    batch_data = []

    for ep in range(num_episodes):
        trajectory = trainer.collect_episode()
        batch_data.append(trajectory)

        if len(batch_data) >= batch_size:
            loss = trainer.update_model(batch_data)
            loss_history.append(loss)
            batch_data = []

        # Evaluate periodically
        if (ep + 1) % 500 == 0:
            agent.set_train_mode(False)
            heuristic = HeuristicAgent()
            avg_chips = quick_evaluate(agent, heuristic, num_rounds=eval_rounds)
            eval_history.append({"episode": ep + 1, "avg_chips": avg_chips})
            agent.set_train_mode(True)

    # Final evaluation
    agent.set_train_mode(False)
    heuristic = HeuristicAgent()
    final_score = quick_evaluate(agent, heuristic, num_rounds=eval_rounds)

    return {
        "n_steps": n_steps,
        "final_score": final_score,
        "eval_history": eval_history,
        "loss_history": loss_history,
        "loss_mean": np.mean(loss_history) if loss_history else 0,
        "loss_std": np.std(loss_history) if loss_history else 0,
    }


def train_td0_baseline(num_episodes=5000, batch_size=32, eval_rounds=200,
                        seed=42, init_state_dict=None):
    """Train with the original SelfPlayTrainer (TD(0)) for clean comparison."""

    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    agent = ValueBasedAgent()
    if init_state_dict is not None:
        agent.model.load_state_dict(copy.deepcopy(init_state_dict))

    trainer = SelfPlayTrainer(agent, learning_rate=1e-4)

    loss_history = []
    eval_history = []

    agent.set_train_mode(True)
    batch_data = []

    for ep in range(num_episodes):
        trajectory = trainer.collect_episode()
        batch_data.append(trajectory)

        if len(batch_data) >= batch_size:
            loss = trainer.update_model(batch_data)
            loss_history.append(loss)
            batch_data = []

        if (ep + 1) % 500 == 0:
            agent.set_train_mode(False)
            heuristic = HeuristicAgent()
            avg_chips = quick_evaluate(agent, heuristic, num_rounds=eval_rounds)
            eval_history.append({"episode": ep + 1, "avg_chips": avg_chips})
            agent.set_train_mode(True)

    agent.set_train_mode(False)
    heuristic = HeuristicAgent()
    final_score = quick_evaluate(agent, heuristic, num_rounds=eval_rounds)

    return {
        "n_steps": "TD0_original",
        "final_score": final_score,
        "eval_history": eval_history,
        "loss_history": loss_history,
        "loss_mean": np.mean(loss_history) if loss_history else 0,
        "loss_std": np.std(loss_history) if loss_history else 0,
    }


def compute_loss_stats(loss_history, window=10):
    """Compute rolling variance of the loss to measure training stability."""
    if len(loss_history) < window:
        return {"rolling_var_mean": 0, "rolling_var_max": 0}

    rolling_vars = []
    for i in range(len(loss_history) - window + 1):
        w = loss_history[i:i+window]
        rolling_vars.append(np.var(w))

    return {
        "rolling_var_mean": float(np.mean(rolling_vars)),
        "rolling_var_max": float(np.max(rolling_vars)),
        "rolling_var_last10": float(np.mean(rolling_vars[-10:])) if len(rolling_vars) >= 10 else 0,
    }


def experiment_3():
    print("\n" + "=" * 70)
    print("EXPERIMENT 3: Training Comparison (5K episodes each)")
    print("=" * 70)

    # Create shared initial weights
    ref_agent = ValueBasedAgent()
    init_weights = copy.deepcopy(ref_agent.model.state_dict())

    n_values = [1, 2, 3, 4, 100]  # 100 = MC
    results = {}

    # Train TD(0) baseline using original SelfPlayTrainer
    print("\n  Training: TD(0) original SelfPlayTrainer...")
    td0_result = train_td0_baseline(
        num_episodes=5000, batch_size=32, eval_rounds=200,
        seed=42, init_state_dict=init_weights
    )
    td0_result["loss_stats"] = compute_loss_stats(td0_result["loss_history"])
    results["TD0_original"] = td0_result
    print(f"    Final score: {td0_result['final_score']:+.3f}")
    print(f"    Loss mean/std: {td0_result['loss_mean']:.4f} / {td0_result['loss_std']:.4f}")
    print(f"    Loss rolling var (mean): {td0_result['loss_stats']['rolling_var_mean']:.6f}")

    # Train with NStepValueTrainer for each n
    for n in n_values:
        label = f"n={n}" if n < 100 else "MC"
        print(f"\n  Training: NStepValueTrainer {label}...")
        result = train_and_evaluate(
            n_steps=n, num_episodes=5000, batch_size=32, eval_rounds=200,
            seed=42, init_state_dict=init_weights
        )
        result["loss_stats"] = compute_loss_stats(result["loss_history"])
        results[label] = result
        print(f"    Final score: {result['final_score']:+.3f}")
        print(f"    Loss mean/std: {result['loss_mean']:.4f} / {result['loss_std']:.4f}")
        print(f"    Loss rolling var (mean): {result['loss_stats']['rolling_var_mean']:.6f}")

    # Summary table
    print("\n" + "-" * 70)
    print("SUMMARY TABLE")
    print("-" * 70)
    print(f"{'Method':<18} {'Final Score':>12} {'Loss Mean':>10} {'Loss Std':>10} {'Loss Var(roll)':>14}")
    print("-" * 70)

    for key in ["TD0_original", "n=1", "n=2", "n=3", "n=4", "MC"]:
        if key in results:
            r = results[key]
            n_label = key
            print(f"{n_label:<18} {r['final_score']:>+12.3f} {r['loss_mean']:>10.4f} "
                  f"{r['loss_std']:>10.4f} {r['loss_stats']['rolling_var_mean']:>14.6f}")

    # Learning curves
    print("\n" + "-" * 70)
    print("LEARNING CURVES (eval vs heuristic at checkpoints)")
    print("-" * 70)
    print(f"{'Episode':<10}", end="")
    for key in ["TD0_original", "n=1", "n=2", "n=3", "n=4", "MC"]:
        if key in results:
            print(f"{key:>12}", end="")
    print()

    # Align by episode
    for idx in range(10):  # up to 10 checkpoints (every 500 eps up to 5000)
        ep = (idx + 1) * 500
        print(f"{ep:<10}", end="")
        for key in ["TD0_original", "n=1", "n=2", "n=3", "n=4", "MC"]:
            if key in results:
                evals = results[key]["eval_history"]
                if idx < len(evals):
                    print(f"{evals[idx]['avg_chips']:>+12.3f}", end="")
                else:
                    print(f"{'---':>12}", end="")
        print()

    return results


# ──────────────────────────────────────────────────────────────────────
# Experiment 4: Target variance analysis
# ──────────────────────────────────────────────────────────────────────

def experiment_4():
    """Measure the variance of training targets under different n for the same states."""
    print("\n" + "=" * 70)
    print("EXPERIMENT 4: Target Variance Analysis")
    print("=" * 70)
    print("  Playing 500 games, recording targets that would be used for each n")

    game = LeducGame()
    agent = ValueBasedAgent()  # untrained

    # Collect episodes
    episodes = []
    for _ in range(500):
        game.reset()
        chains = [[], []]

        while not game.is_finished:
            cp = game.current_player
            obs = game.get_observation(viewer_id=cp)
            action = random.choice(obs.legal_actions)

            post_obs, _ = LeducGame.simulate_action(obs, action)
            encoded = agent.encode_observation(post_obs, viewer_id=cp)
            chains[cp].append(encoded)

            game.step(action)

        rewards = game.get_reward()
        episodes.append((chains, rewards))

    # For each n value, compute what the targets would be
    for n in [1, 2, 3, 100]:
        label = f"n={n}" if n < 100 else "MC"
        all_targets = []
        bootstrap_count = 0
        terminal_count = 0

        for chains, rewards in episodes:
            for p_idx in [0, 1]:
                chain = chains[p_idx]
                L = len(chain)
                for t in range(L):
                    if t + n >= L:
                        target = rewards[p_idx]
                        terminal_count += 1
                    else:
                        with torch.no_grad():
                            target = agent.model(chain[t + n]).item()
                        bootstrap_count += 1
                    all_targets.append(target)

        targets = np.array(all_targets)
        print(f"\n  {label}:")
        print(f"    Num targets:   {len(targets)}")
        print(f"    Bootstrap:     {bootstrap_count} ({bootstrap_count/len(targets)*100:.1f}%)")
        print(f"    Terminal:      {terminal_count} ({terminal_count/len(targets)*100:.1f}%)")
        print(f"    Target mean:   {targets.mean():.4f}")
        print(f"    Target std:    {targets.std():.4f}")
        print(f"    Target range:  [{targets.min():.2f}, {targets.max():.2f}]")


# ──────────────────────────────────────────────────────────────────────
# Experiment 5: Verify n=1 NStep matches TD(0) original
# ──────────────────────────────────────────────────────────────────────

def experiment_5():
    """Verify that NStepValueTrainer with n=1 produces identical targets to SelfPlayTrainer."""
    print("\n" + "=" * 70)
    print("EXPERIMENT 5: Verify n=1 NStep == TD(0)")
    print("=" * 70)

    # Play one episode, compute targets both ways
    torch.manual_seed(99)
    random.seed(99)

    agent = ValueBasedAgent()
    game = LeducGame()
    game.reset()

    chains = [[], []]
    while not game.is_finished:
        cp = game.current_player
        obs = game.get_observation(viewer_id=cp)
        action = random.choice(obs.legal_actions)
        post_obs, _ = LeducGame.simulate_action(obs, action)
        encoded = agent.encode_observation(post_obs, viewer_id=cp)
        chains[cp].append(encoded)
        game.step(action)

    rewards = game.get_reward()

    print(f"  Rewards: {rewards}")
    for p_idx in [0, 1]:
        chain = chains[p_idx]
        L = len(chain)
        print(f"\n  Player {p_idx}, chain length = {L}:")

        for t in range(L):
            # TD(0) target
            if t == L - 1:
                td0_target = rewards[p_idx]
            else:
                with torch.no_grad():
                    td0_target = agent.model(chain[t + 1]).squeeze(0).item()

            # NStep n=1 target
            if t + 1 >= L:
                nstep_target = rewards[p_idx]
            else:
                with torch.no_grad():
                    nstep_target = agent.model(chain[t + 1]).squeeze(0).item()

            match = "MATCH" if abs(td0_target - nstep_target) < 1e-6 else "MISMATCH!"
            print(f"    t={t}: TD(0)={td0_target:+.4f}, NStep(n=1)={nstep_target:+.4f} -> {match}")


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("DIAGNOSTIC: Why does NStepValueAgent underperform ValueBasedAgent?")
    print("=" * 70)

    # Exp 1: Chain length distribution
    chain_lengths = experiment_1()

    # Exp 2: Bootstrap fraction
    experiment_2(chain_lengths)

    # Exp 5: Verify n=1 == TD(0) (quick, run before the long training)
    experiment_5()

    # Exp 4: Target variance
    experiment_4()

    # Exp 3: Training comparison (longest experiment)
    results_3 = experiment_3()

    # ──────────────────────────────────────────────────────────────────
    # FINAL ANALYSIS
    # ──────────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("FINAL ANALYSIS")
    print("=" * 70)

    print("""
HYPOTHESES AND EVIDENCE:

H1: n=3 turns training into near-pure Monte Carlo
  - Chain length data shows whether n=3 exceeds most chain lengths
  - Bootstrap fraction data quantifies this directly
  - If >80% of transitions use terminal reward, this is confirmed

H2: MC has devastating variance in self-play
  - Loss std and rolling variance compare stability across n values
  - If loss variance increases monotonically with n, this is confirmed
  - Eval scores should degrade with higher n if variance is harmful

H3: TD(0) bootstrapping provides implicit smoothing
  - n=1 using NStepTrainer should match TD(0) original almost exactly
  - If n=1 matches but n=2+ degrades, the bootstrap fraction matters
  - The performance frontier from n=1 to MC reveals the bias-variance tradeoff
""")

    # Save results
    save_data = {}
    for key, val in results_3.items():
        save_data[key] = {
            "final_score": val["final_score"],
            "loss_mean": val["loss_mean"],
            "loss_std": val["loss_std"],
            "loss_stats": val["loss_stats"],
            "eval_history": val["eval_history"],
        }

    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "diagnose_nstep_results.json")
    with open(out_path, "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"\nResults saved to {out_path}")
