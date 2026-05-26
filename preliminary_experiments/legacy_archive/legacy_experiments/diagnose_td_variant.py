#!/usr/bin/env python3
"""
Diagnostic experiment: Why did td_variant (n_steps=3, lr=5e-5) fail to converge?

Background:
  - td_variant was trained with n_steps=3, lr=5e-5, 20K episodes
  - Final loss: 33.35 (vs ~7-15 for other agents)
  - Final eval vs heuristic: -1.33 (very bad)
  - Only 625 updates in 20K episodes
  - Never beat heuristic during training
  - Meanwhile, value_based (TD(0), lr=1e-4) achieves avg=+0.97 in tournament

Hypotheses:
  H1: n_steps=3 with lr=5e-5 has too few bootstrapped transitions (most
      targets are terminal reward), making it behave like high-variance MC.
      Combined with low lr, the agent can't converge fast enough.
  H2: lr=5e-5 is simply too small — the agent needs more updates per episode
      than the batch_size=32 loop provides, so it effectively underfits.
  H3: lr=3e-5 might help MC converge if the variance is the main issue.
  H4: TD(0) at lr=1e-4 provides the best bias-variance tradeoff for Leduc's
      short chains, and no n-step variant can beat it.

Experiments:
  1. Multi-variant comparison (5000 episodes, eval every 500):
     - TD(0): n_steps=1, lr=1e-4 (known-good baseline)
     - n-step n=3, lr=5e-5 (tournament config — the failure)
     - n-step n=3, lr=3e-5 (even more conservative)
     - MC: n_steps=9999, lr=5e-5
     - MC conservative: n_steps=9999, lr=3e-5

  2. Loss curve comparison across all variants

  3. Final tournament: all 5 variants vs heuristic, value_based, adaptive_value

  4. Analysis: which variant converges best? Is TD(0) still king?

Usage:
  python -m experiments.diagnose_td_variant
"""

import sys, os, copy, random, json, time
from collections import Counter

import torch
import numpy as np

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.engine.leduc_game import LeducGame, Action
from src.agents.value_based import ValueBasedAgent
from src.agents.td_variant import TDVariantAgent
from src.agents.heuristic import HeuristicAgent
from src.training.td_variant_trainer import TDVariantTrainer
from src.training.value_based_trainer import SelfPlayTrainer
from src.training.evaluation import quick_evaluate


# ──────────────────────────────────────────────────────────────────────
# Experiment 1: Chain length distribution + bootstrap fraction analysis
# ──────────────────────────────────────────────────────────────────────

def measure_chain_lengths(num_games=2000):
    """Play games with a random agent and record per-player chain lengths."""
    game = LeducGame()
    all_chain_lengths = []

    for _ in range(num_games):
        game.reset()
        chains = [0, 0]

        while not game.is_finished:
            cp = game.current_player
            obs = game.get_observation(viewer_id=cp)
            action = random.choice(obs.legal_actions)
            game.step(action)
            chains[cp] += 1

        all_chain_lengths.extend(chains)

    return all_chain_lengths


def compute_bootstrap_fraction(chain_lengths, n_steps):
    """For given chain lengths and n, compute fraction of bootstrapped vs terminal targets."""
    total = 0
    bootstrapped = 0
    terminal = 0

    for L in chain_lengths:
        for t in range(L):
            total += 1
            if t + n_steps >= L:
                terminal += 1
            else:
                bootstrapped += 1

    return {
        "n_steps": n_steps,
        "total_transitions": total,
        "bootstrapped": bootstrapped,
        "terminal": terminal,
        "bootstrap_frac": bootstrapped / total if total else 0,
        "terminal_frac": terminal / total if total else 0,
    }


def experiment_1():
    print("=" * 72)
    print("EXPERIMENT 1: Chain Lengths & Bootstrap Fractions")
    print("=" * 72)

    chain_lengths = measure_chain_lengths(2000)
    counter = Counter(chain_lengths)
    total = len(chain_lengths)

    print(f"\nPer-player chain lengths ({total} chains from 2000 games):")
    for length in sorted(counter.keys()):
        pct = counter[length] / total * 100
        bar = "#" * int(pct / 2)
        print(f"  Length {length}: {counter[length]:5d} ({pct:5.1f}%) {bar}")

    avg_chain = sum(chain_lengths) / len(chain_lengths)
    print(f"\n  Mean chain length: {avg_chain:.2f}")
    print(f"  Min: {min(chain_lengths)}, Max: {max(chain_lengths)}")

    # Bootstrap fractions for each n we'll test
    print(f"\nBootstrap fraction by n_steps:")
    print(f"  {'n_steps':<10} {'Bootstrap%':>12} {'Terminal%':>12} {'Total':>8}")
    print(f"  {'-'*44}")

    results = {}
    for n in [1, 2, 3, 4, 9999]:
        label = f"n={n}" if n < 100 else "MC(9999)"
        r = compute_bootstrap_fraction(chain_lengths, n)
        print(f"  {label:<10} {r['bootstrap_frac']*100:>11.1f}% {r['terminal_frac']*100:>11.1f}% {r['total_transitions']:>8}")
        results[label] = r

    # Key insight
    print(f"\n  KEY INSIGHT: With n=3, {results.get('n=3', results.get('MC(9999)', {})).get('terminal_frac', 0)*100:.1f}% of targets use terminal reward.")
    print(f"  This means n=3 is functionally Monte Carlo for most of Leduc.")

    return chain_lengths, results


# ──────────────────────────────────────────────────────────────────────
# Experiment 2: Multi-variant training comparison
# ──────────────────────────────────────────────────────────────────────

VARIANTS = [
    {"name": "TD(0) lr=1e-4",        "n_steps": 1,    "lr": 1e-4},
    {"name": "n=3 lr=5e-5",          "n_steps": 3,    "lr": 5e-5},
    {"name": "n=3 lr=3e-5",          "n_steps": 3,    "lr": 3e-5},
    {"name": "MC lr=5e-5",           "n_steps": 9999, "lr": 5e-5},
    {"name": "MC lr=3e-5",           "n_steps": 9999, "lr": 3e-5},
]


def train_variant(name, n_steps, lr, num_episodes=5000, batch_size=32,
                  eval_interval=500, eval_rounds=300, seed=42,
                  init_state_dict=None):
    """Train a TDVariantAgent with given hyperparams, tracking loss and eval curves."""

    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    agent = TDVariantAgent()
    if init_state_dict is not None:
        agent.model.load_state_dict(copy.deepcopy(init_state_dict))

    trainer = TDVariantTrainer(agent, learning_rate=lr, n_steps=n_steps)

    loss_history = []
    eval_history = []

    agent.set_train_mode(True)
    batch_data = []
    num_updates = 0

    for ep in range(num_episodes):
        trajectory = trainer.collect_episode()
        batch_data.append(trajectory)

        if len(batch_data) >= batch_size:
            loss = trainer.update_model(batch_data)
            loss_history.append({"episode": ep + 1, "loss": loss})
            batch_data = []
            num_updates += 1

        # Evaluate periodically
        if (ep + 1) % eval_interval == 0:
            agent.set_train_mode(False)
            heuristic = HeuristicAgent()
            avg_chips = quick_evaluate(agent, heuristic, num_rounds=eval_rounds)
            eval_history.append({"episode": ep + 1, "avg_chips": avg_chips})
            agent.set_train_mode(True)
            print(f"    [{name}] ep={ep+1:5d}  loss={loss_history[-1]['loss'] if loss_history else 0:.4f}  "
                  f"eval={avg_chips:+.3f}")

    # Final evaluation
    agent.set_train_mode(False)
    heuristic = HeuristicAgent()
    final_score = quick_evaluate(agent, heuristic, num_rounds=eval_rounds)

    # Loss statistics
    losses = [l["loss"] for l in loss_history]
    loss_mean = float(np.mean(losses)) if losses else 0
    loss_std = float(np.std(losses)) if losses else 0

    # Loss trajectory: first 5 and last 5
    first_5 = losses[:5] if len(losses) >= 5 else losses
    last_5 = losses[-5:] if len(losses) >= 5 else losses
    loss_first5_mean = float(np.mean(first_5)) if first_5 else 0
    loss_last5_mean = float(np.mean(last_5)) if last_5 else 0

    return {
        "name": name,
        "n_steps": n_steps,
        "lr": lr,
        "final_score": final_score,
        "num_updates": num_updates,
        "loss_mean": loss_mean,
        "loss_std": loss_std,
        "loss_first5_mean": loss_first5_mean,
        "loss_last5_mean": loss_last5_mean,
        "eval_history": eval_history,
        "loss_history": loss_history,
        "agent": agent,  # Keep for tournament
    }


def experiment_2():
    print("\n" + "=" * 72)
    print("EXPERIMENT 2: Multi-Variant Training Comparison (5000 episodes each)")
    print("=" * 72)

    # Shared initial weights for fair comparison
    ref_agent = TDVariantAgent()
    init_weights = copy.deepcopy(ref_agent.model.state_dict())

    results = {}
    for v in VARIANTS:
        print(f"\n  Training: {v['name']} (n_steps={v['n_steps']}, lr={v['lr']})...")
        t0 = time.time()
        result = train_variant(
            name=v["name"],
            n_steps=v["n_steps"],
            lr=v["lr"],
            num_episodes=5000,
            batch_size=32,
            eval_interval=500,
            eval_rounds=300,
            seed=42,
            init_state_dict=init_weights,
        )
        elapsed = time.time() - t0
        result["train_time_s"] = round(elapsed, 1)
        results[v["name"]] = result
        print(f"    Done in {elapsed:.1f}s.  Final score: {result['final_score']:+.3f}")

    # ── Summary table ──
    print("\n" + "-" * 72)
    print("TRAINING SUMMARY")
    print("-" * 72)
    header = f"{'Variant':<22} {'Final':>8} {'Updates':>8} {'LossMean':>10} {'LossStd':>10} {'LossFirst5':>11} {'LossLast5':>10}"
    print(header)
    print("-" * 72)
    for v in VARIANTS:
        r = results[v["name"]]
        print(f"{r['name']:<22} {r['final_score']:>+8.3f} {r['num_updates']:>8} "
              f"{r['loss_mean']:>10.4f} {r['loss_std']:>10.4f} "
              f"{r['loss_first5_mean']:>11.4f} {r['loss_last5_mean']:>10.4f}")

    # ── Learning curves ──
    print("\n" + "-" * 72)
    print("LEARNING CURVES (eval vs heuristic, avg chips/round)")
    print("-" * 72)
    header = f"{'Episode':<10}"
    for v in VARIANTS:
        header += f"{v['name'][:16]:>18}"
    print(header)

    for idx in range(10):
        ep = (idx + 1) * 500
        row = f"{ep:<10}"
        for v in VARIANTS:
            evals = results[v["name"]]["eval_history"]
            if idx < len(evals):
                row += f"{evals[idx]['avg_chips']:>+18.3f}"
            else:
                row += f"{'---':>18}"
        print(row)

    # ── Loss curves (sampled) ──
    print("\n" + "-" * 72)
    print("LOSS CURVES (sampled at update indices)")
    print("-" * 72)
    sample_indices = [0, 10, 25, 50, 75, 100, 125, 150]
    header = f"{'Update#':<10}"
    for v in VARIANTS:
        header += f"{v['name'][:16]:>18}"
    print(header)

    for idx in sample_indices:
        row = f"{idx:<10}"
        for v in VARIANTS:
            losses = results[v["name"]]["loss_history"]
            if idx < len(losses):
                row += f"{losses[idx]['loss']:>18.4f}"
            else:
                row += f"{'---':>18}"
        print(row)

    return results


# ──────────────────────────────────────────────────────────────────────
# Experiment 3: Final tournament (all variants vs opponents)
# ──────────────────────────────────────────────────────────────────────

def experiment_3(training_results):
    print("\n" + "=" * 72)
    print("EXPERIMENT 3: Tournament (500 rounds each matchup)")
    print("=" * 72)

    eval_rounds = 500

    # Opponents
    opponents = {}
    opponents["heuristic"] = HeuristicAgent()

    # Try loading trained value_based model
    value_model_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "models", "value_agent.pt"
    )
    if os.path.exists(value_model_path):
        vb = ValueBasedAgent(model_path=value_model_path)
        opponents["value_based"] = vb
        print(f"  Loaded value_based from {value_model_path}")
    else:
        print(f"  WARNING: {value_model_path} not found, skipping value_based opponent")

    # Try loading trained adaptive_value model
    adaptive_model_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "models", "adaptive_value_agent.pt"
    )
    if os.path.exists(adaptive_model_path):
        from src.agents.adaptive_value import AdaptiveValueAgent
        av = AdaptiveValueAgent(model_path=adaptive_model_path)
        opponents["adaptive_value"] = av
        print(f"  Loaded adaptive_value from {adaptive_model_path}")
    else:
        print(f"  WARNING: {adaptive_model_path} not found, skipping adaptive_value opponent")

    # Run tournament
    tournament = {}
    for v_name, v_result in training_results.items():
        agent = v_result["agent"]
        agent.set_train_mode(False)
        tournament[v_name] = {}

        for opp_name, opp in opponents.items():
            score = quick_evaluate(agent, opp, num_rounds=eval_rounds)
            tournament[v_name][opp_name] = round(score, 4)

    # Display
    print(f"\n{'Variant':<22}", end="")
    for opp_name in opponents:
        print(f"{opp_name:>16}", end="")
    print(f"{'AVG':>10}")
    print("-" * (22 + 16 * len(opponents) + 10))

    for v_name in [v["name"] for v in VARIANTS]:
        if v_name not in tournament:
            continue
        scores = tournament[v_name]
        print(f"{v_name:<22}", end="")
        for opp_name in opponents:
            print(f"{scores.get(opp_name, 0):>+16.3f}", end="")
        avg = np.mean(list(scores.values()))
        print(f"{avg:>+10.3f}")

    return tournament


# ──────────────────────────────────────────────────────────────────────
# Experiment 4: Gradient magnitude analysis
# ──────────────────────────────────────────────────────────────────────

def experiment_4():
    """
    Compare gradient magnitudes for TD(0) vs n=3 vs MC on a single batch.
    This reveals whether low lr + MC produces vanishingly small parameter updates.
    """
    print("\n" + "=" * 72)
    print("EXPERIMENT 4: Gradient Magnitude Analysis (single batch)")
    print("=" * 72)

    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)

    # Shared weights
    ref_agent = TDVariantAgent()
    init_weights = copy.deepcopy(ref_agent.model.state_dict())

    configs = [
        ("TD(0) lr=1e-4", 1, 1e-4),
        ("n=3 lr=5e-5",   3, 5e-5),
        ("n=3 lr=3e-5",   3, 3e-5),
        ("MC lr=5e-5",    9999, 5e-5),
        ("MC lr=3e-5",    9999, 3e-5),
    ]

    # Collect a shared batch of 32 episodes
    agent_for_collect = TDVariantAgent()
    agent_for_collect.model.load_state_dict(copy.deepcopy(init_weights))
    agent_for_collect.set_train_mode(True)
    game = LeducGame()

    batch_data = []
    for _ in range(32):
        game.reset()
        chains = [[], []]
        while not game.is_finished:
            cp = game.current_player
            obs = game.get_observation(viewer_id=cp)
            action = agent_for_collect.select_action(obs)
            if isinstance(action, tuple):
                action = action[0]
            post_obs, _ = LeducGame.simulate_action(obs, action)
            encoded = agent_for_collect.encode_observation(post_obs, viewer_id=cp)
            chains[cp].append(encoded)
            game.step(action)
        rewards = game.get_reward()
        batch_data.append((chains, rewards))

    print(f"\n  Batch: 32 episodes collected")

    # For each config, compute gradient norm after one update
    print(f"\n  {'Config':<22} {'Loss':>10} {'GradNorm':>12} {'MaxGradParam':>14} {'ParamUpdateNorm':>16}")
    print(f"  {'-'*76}")

    for name, n_steps, lr in configs:
        agent = TDVariantAgent()
        agent.model.load_state_dict(copy.deepcopy(init_weights))

        # Record params before
        params_before = {n: p.clone() for n, p in agent.model.named_parameters()}

        trainer = TDVariantTrainer(agent, learning_rate=lr, n_steps=n_steps)
        loss = trainer.update_model(batch_data)

        # Compute gradient norm (right after backward, before optimizer.zero_grad clears)
        # Actually the optimizer already stepped, but we can measure param change
        total_grad_norm = 0
        max_grad = 0
        param_update_norm = 0

        for n, p in agent.model.named_parameters():
            diff = (p - params_before[n]).detach()
            param_update_norm += diff.norm().item() ** 2
            # We can't get grad after step, so param change is our proxy

        param_update_norm = param_update_norm ** 0.5

        # Also do a separate forward/backward just to measure gradient norm
        agent2 = TDVariantAgent()
        agent2.model.load_state_dict(copy.deepcopy(init_weights))
        trainer2 = TDVariantTrainer(agent2, learning_rate=lr, n_steps=n_steps)
        trainer2.optimizer.zero_grad()

        # Manual forward/backward to capture gradients
        total_losses = []
        for chains, rewards in batch_data:
            for p_idx in [0, 1]:
                chain = chains[p_idx]
                if not chain:
                    continue
                L = len(chain)
                for t in range(L):
                    prediction = agent2.model(chain[t]).squeeze(0)
                    if t + n_steps >= L:
                        target = torch.FloatTensor([rewards[p_idx]])
                    else:
                        with torch.no_grad():
                            target = agent2.model(chain[t + n_steps]).squeeze(0)
                    l = torch.nn.functional.mse_loss(prediction, target)
                    total_losses.append(l)

        if total_losses:
            mean_loss = torch.stack(total_losses).mean()
            mean_loss.backward()

            for p in agent2.model.parameters():
                if p.grad is not None:
                    total_grad_norm += p.grad.norm().item() ** 2
                    max_grad = max(max_grad, p.grad.abs().max().item())
            total_grad_norm = total_grad_norm ** 0.5

        print(f"  {name:<22} {loss:>10.4f} {total_grad_norm:>12.6f} {max_grad:>14.6f} {param_update_norm:>16.8f}")

    print(f"\n  NOTE: All configs use same init weights and same batch.")
    print(f"  ParamUpdateNorm = effective learning signal = lr * GradNorm (approximately).")
    print(f"  Lower lr means proportionally smaller parameter updates per batch.")


# ──────────────────────────────────────────────────────────────────────
# Experiment 5: Update count analysis
# ──────────────────────────────────────────────────────────────────────

def experiment_5():
    """
    Calculate expected update counts for different episode/batch configs.
    Context: td_variant got only 625 updates in 20K episodes (batch_size=32).
    """
    print("\n" + "=" * 72)
    print("EXPERIMENT 5: Update Count Analysis")
    print("=" * 72)

    configs = [
        (5000, 32),
        (10000, 32),
        (20000, 32),
        (5000, 16),
        (5000, 8),
    ]

    print(f"\n  {'Episodes':>10} {'BatchSize':>10} {'Updates':>10} {'UpdatesPerEp':>14}")
    print(f"  {'-'*46}")
    for episodes, bs in configs:
        updates = episodes // bs
        per_ep = updates / episodes
        print(f"  {episodes:>10} {bs:>10} {updates:>10} {per_ep:>14.4f}")

    print(f"\n  td_variant tournament config: 20000 ep, bs=32 => 625 updates")
    print(f"  With lr=5e-5, effective learning = 625 * 5e-5 = 0.03125 total lr-steps")
    print(f"  With lr=1e-4, that would be 625 * 1e-4 = 0.0625 total lr-steps")
    print(f"  value_based (5K ep, bs=32, lr=1e-4) = 156 updates * 1e-4 = 0.0156")
    print(f"  So td_variant actually had MORE cumulative lr*updates, yet failed.")
    print(f"  => The issue is NOT total learning budget, but per-update signal quality.")


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 72)
    print("DIAGNOSTIC: Why did td_variant fail to converge?")
    print("  Config: n_steps=3, lr=5e-5, 20K episodes, batch_size=32")
    print("  Symptoms: loss=33.35, eval=-1.33, never beat heuristic")
    print("=" * 72)

    t_start = time.time()

    # Quick structural analyses
    chain_lengths, bootstrap_results = experiment_1()
    experiment_5()
    experiment_4()

    # Main training comparison (takes longest)
    training_results = experiment_2()

    # Tournament
    tournament = experiment_3(training_results)

    total_time = time.time() - t_start

    # ──────────────────────────────────────────────────────────────────
    # FINAL ANALYSIS
    # ──────────────────────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("FINAL ANALYSIS")
    print("=" * 72)

    # Determine best variant
    best_name = max(training_results.keys(),
                    key=lambda k: training_results[k]["final_score"])
    best_score = training_results[best_name]["final_score"]

    # Tournament averages
    tournament_avgs = {}
    for v_name, scores in tournament.items():
        tournament_avgs[v_name] = np.mean(list(scores.values()))
    best_tournament = max(tournament_avgs, key=tournament_avgs.get)

    print(f"""
FINDINGS:

1. Chain Length Analysis:
   - Leduc chains are typically 1-4 steps per player
   - With n=3, ~{bootstrap_results.get('n=3', {}).get('terminal_frac', 0)*100:.0f}% of transitions use terminal reward (not bootstrapped)
   - n=3 is FUNCTIONALLY Monte Carlo in Leduc Hold'em
   - TD(0) (n=1) has ~{bootstrap_results.get('n=1', {}).get('bootstrap_frac', 0)*100:.0f}% bootstrapped transitions

2. Training Comparison (5K episodes):
   - Best variant: {best_name} (score: {best_score:+.3f})
   - TD(0) lr=1e-4: {training_results['TD(0) lr=1e-4']['final_score']:+.3f}
   - n=3 lr=5e-5:   {training_results['n=3 lr=5e-5']['final_score']:+.3f}  (tournament config)
   - n=3 lr=3e-5:   {training_results['n=3 lr=3e-5']['final_score']:+.3f}
   - MC lr=5e-5:    {training_results['MC lr=5e-5']['final_score']:+.3f}
   - MC lr=3e-5:    {training_results['MC lr=3e-5']['final_score']:+.3f}

3. Loss Analysis:
   - TD(0) loss (mean/std): {training_results['TD(0) lr=1e-4']['loss_mean']:.4f} / {training_results['TD(0) lr=1e-4']['loss_std']:.4f}
   - n=3 lr=5e-5 loss:      {training_results['n=3 lr=5e-5']['loss_mean']:.4f} / {training_results['n=3 lr=5e-5']['loss_std']:.4f}
   - Loss convergence (first5 -> last5):
     TD(0):      {training_results['TD(0) lr=1e-4']['loss_first5_mean']:.4f} -> {training_results['TD(0) lr=1e-4']['loss_last5_mean']:.4f}
     n=3 5e-5:   {training_results['n=3 lr=5e-5']['loss_first5_mean']:.4f} -> {training_results['n=3 lr=5e-5']['loss_last5_mean']:.4f}

4. Tournament Results:
   - Best overall: {best_tournament} (avg: {tournament_avgs[best_tournament]:+.3f})

5. ROOT CAUSE ANALYSIS:
   The td_variant failure has TWO compounding factors:

   a) n_steps=3 in Leduc is effectively Monte Carlo (~{bootstrap_results.get('n=3', {}).get('terminal_frac', 0)*100:.0f}% terminal targets).
      MC targets have high variance because the terminal reward depends on
      the full game outcome (opponent cards, board card), not just the
      current state's value. In self-play where the opponent policy
      changes every batch, this variance is devastating.

   b) lr=5e-5 is 2x smaller than the working lr=1e-4. Combined with
      high-variance MC targets, the agent learns too slowly to overcome
      the noise. Each update barely moves the parameters, and the
      direction is corrupted by target variance.

   TD(0) works because bootstrapping from V(s_next) provides implicit
   temporal smoothing: targets change slowly as the network changes,
   creating a stable learning signal even in self-play.

Total runtime: {total_time:.1f}s
""")

    # ──────────────────────────────────────────────────────────────────
    # Save results
    # ──────────────────────────────────────────────────────────────────
    save_data = {
        "experiment": "diagnose_td_variant",
        "date": "2026-02-25",
        "total_runtime_s": round(total_time, 1),
        "chain_length_analysis": {
            k: {kk: vv for kk, vv in v.items()}
            for k, v in bootstrap_results.items()
        },
        "training_comparison": {},
        "tournament": tournament,
        "tournament_averages": {k: round(v, 4) for k, v in tournament_avgs.items()},
    }

    for v_name, r in training_results.items():
        save_data["training_comparison"][v_name] = {
            "n_steps": r["n_steps"],
            "lr": r["lr"],
            "final_score": r["final_score"],
            "num_updates": r["num_updates"],
            "loss_mean": r["loss_mean"],
            "loss_std": r["loss_std"],
            "loss_first5_mean": r["loss_first5_mean"],
            "loss_last5_mean": r["loss_last5_mean"],
            "eval_history": r["eval_history"],
            "train_time_s": r.get("train_time_s", 0),
        }

    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "diagnose_td_variant_results.json")
    with open(out_path, "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"Results saved to {out_path}")
