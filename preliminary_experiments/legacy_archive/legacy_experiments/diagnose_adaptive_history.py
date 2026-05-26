"""
Diagnose why AdaptiveHistoryAgent underperforms AdaptiveValueAgent.

Hypotheses:
  H1: Network capacity — wider network (128 hidden, ~4x params) needs more data to converge.
  H2: History features are noise in Leduc — the network can't learn to ignore 16 mostly-zero dims.
  H3: Action history encoding correctness — are features carried forward properly?
  H4: Gradient dynamics — wider network changes optimization landscape harmfully.

Experiments:
  Exp 1: AdaptiveValue with WIDER network (128 hidden, 19 dims) — isolates network size effect.
  Exp 2: AdaptiveHistory trained 2x longer (1334 sessions) — tests if it just needs more data.
  Exp 3: Parameter count comparison — quantify capacity difference.
  Exp 4: History feature analysis — sample actual feature vectors during training to check informativeness.
  Exp 5: Loss curve comparison — compare training dynamics between all variants.
"""

import os
import sys
import time
import math
import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.agents.adaptive_value import AdaptiveValueAgent
from src.agents.adaptive_history import AdaptiveHistoryAgent
from src.agents.value_based import ValueNetwork
from src.training.adaptive_trainer import AdaptiveTrainer
from src.training.adaptive_history_trainer import AdaptiveHistoryTrainer
from src.training.evaluation import evaluate_agents, quick_evaluate
from src.agents.heuristic import HeuristicAgent
from src.engine.poker_session import PokerSession


# ──────────────────────────────────────────────
# Experiment 3: Parameter Count Analysis
# ──────────────────────────────────────────────

def exp3_parameter_counts():
    """Count and compare parameters for each architecture."""
    print("\n" + "=" * 60)
    print("  EXPERIMENT 3: Parameter Count Analysis")
    print("=" * 60 + "\n")

    configs = {
        "AdaptiveValue (19->64->64->1)": (19, 64),
        "AdaptiveHistory (35->128->128->1)": (35, 128),
        "AdaptiveValue-Wide (19->128->128->1)": (19, 128),
        "AdaptiveHistory-Narrow (35->64->64->1)": (35, 64),
    }

    results = {}
    for name, (in_size, hidden) in configs.items():
        net = ValueNetwork(in_size, hidden_size=hidden)
        total = sum(p.numel() for p in net.parameters())
        trainable = sum(p.numel() for p in net.parameters() if p.requires_grad)

        # Break down by layer
        layers = {}
        for pname, p in net.named_parameters():
            layers[pname] = p.numel()

        results[name] = {"total": total, "trainable": trainable, "layers": layers}
        print(f"  {name}")
        print(f"    Total parameters: {total:,}")
        for lname, count in layers.items():
            print(f"      {lname}: {count:,}")
        print()

    # Ratios
    base_params = results["AdaptiveValue (19->64->64->1)"]["total"]
    print(f"  Parameter ratios (relative to AdaptiveValue baseline):")
    for name, r in results.items():
        ratio = r["total"] / base_params
        print(f"    {name}: {ratio:.2f}x ({r['total']:,} params)")

    return results


# ──────────────────────────────────────────────
# Experiment 4: History Feature Informativeness
# ──────────────────────────────────────────────

def exp4_history_feature_analysis(num_sessions=50):
    """Sample actual history features during gameplay to check informativeness."""
    print("\n" + "=" * 60)
    print("  EXPERIMENT 4: History Feature Informativeness Analysis")
    print("=" * 60 + "\n")

    agent = AdaptiveHistoryAgent()
    session = PokerSession()

    all_history_features = []
    all_stats_features = []
    all_base_features = []
    feature_by_step = {"early": [], "mid": [], "late": []}

    for session_idx in range(num_sessions):
        session.reset()
        hands_per_session = 30

        for hand_idx in range(hands_per_session):
            session.new_hand()
            step_count = 0

            while not session.is_finished:
                player = session.current_player
                obs = session.get_observation(viewer_id=player)

                # Encode the full observation
                encoded = agent.encode_observation(obs, viewer_id=player)
                full_vec = encoded.squeeze(0).detach().numpy()

                # Split into components: base[0:15], stats[15:19], history[19:35]
                base_part = full_vec[:15]
                stats_part = full_vec[15:19]
                history_part = full_vec[19:35]

                all_base_features.append(base_part)
                all_stats_features.append(stats_part)
                all_history_features.append(history_part)

                # Categorize by position in session
                if hand_idx < 5:
                    feature_by_step["early"].append(history_part)
                elif hand_idx < 15:
                    feature_by_step["mid"].append(history_part)
                else:
                    feature_by_step["late"].append(history_part)

                # Take an action to advance the game
                action = agent.select_action(obs)
                session.step(action)
                step_count += 1

    # Analyze history features
    history_array = np.array(all_history_features)
    stats_array = np.array(all_stats_features)
    base_array = np.array(all_base_features)

    print(f"  Total observations sampled: {len(all_history_features)}")
    print()

    # Per-feature statistics
    print("  History feature statistics (16 dims):")
    print(f"  {'Dim':>4s}  {'Mean':>8s}  {'Std':>8s}  {'%Zero':>8s}  {'Min':>8s}  {'Max':>8s}  {'Description':>30s}")
    print("  " + "-" * 90)

    history_labels = [
        "R0 player_fold/n", "R0 player_call/n", "R0 player_raise/n",
        "R0 opp_fold/n", "R0 opp_call/n", "R0 opp_raise/n",
        "R0 total_actions/6", "R0 has_raise",
        "R1 player_fold/n", "R1 player_call/n", "R1 player_raise/n",
        "R1 opp_fold/n", "R1 opp_call/n", "R1 opp_raise/n",
        "R1 total_actions/6", "R1 has_raise",
    ]

    zero_rates = []
    stds = []
    for d in range(16):
        col = history_array[:, d]
        mean = col.mean()
        std = col.std()
        pct_zero = (col == 0).mean() * 100
        zero_rates.append(pct_zero)
        stds.append(std)
        label = history_labels[d] if d < len(history_labels) else ""
        print(f"  {d:>4d}  {mean:>8.4f}  {std:>8.4f}  {pct_zero:>7.1f}%  {col.min():>8.4f}  {col.max():>8.4f}  {label:>30s}")

    print()
    avg_zero_rate = np.mean(zero_rates)
    print(f"  Average zero-rate across history dims: {avg_zero_rate:.1f}%")
    print(f"  Average std across history dims: {np.mean(stds):.4f}")

    # Compare with stats features
    print(f"\n  Stats feature statistics (4 dims):")
    stats_labels = ["fold_rate", "raise_rate", "fold_to_raise", "confidence"]
    stats_stds = []
    for d in range(4):
        col = stats_array[:, d]
        mean = col.mean()
        std = col.std()
        pct_zero = (col == 0).mean() * 100
        stats_stds.append(std)
        label = stats_labels[d]
        print(f"  {d:>4d}  {mean:>8.4f}  {std:>8.4f}  {pct_zero:>7.1f}%  {col.min():>8.4f}  {col.max():>8.4f}  {label:>30s}")

    print(f"\n  Average std across stats dims: {np.mean(stats_stds):.4f}")

    # How do history features evolve across the session?
    print(f"\n  History feature evolution within session:")
    for phase, feats in feature_by_step.items():
        if feats:
            arr = np.array(feats)
            nonzero_frac = (arr != 0).mean() * 100
            mean_norm = np.linalg.norm(arr, axis=1).mean()
            print(f"    {phase:>6s}: n={len(feats):>5d}, nonzero_frac={nonzero_frac:.1f}%, mean_L2_norm={mean_norm:.4f}")

    # Signal-to-noise comparison: what fraction of the input vector's variance comes from history?
    full_array = np.concatenate([base_array, stats_array, history_array], axis=1)
    total_var = full_array.var(axis=0).sum()
    base_var = base_array.var(axis=0).sum()
    stats_var = stats_array.var(axis=0).sum()
    history_var = history_array.var(axis=0).sum()

    print(f"\n  Variance budget:")
    print(f"    Total variance (all 35 dims): {total_var:.4f}")
    print(f"    Base features (15 dims): {base_var:.4f} ({base_var/total_var*100:.1f}%)")
    print(f"    Stats features (4 dims): {stats_var:.4f} ({stats_var/total_var*100:.1f}%)")
    print(f"    History features (16 dims): {history_var:.4f} ({history_var/total_var*100:.1f}%)")
    print(f"    History variance per dim: {history_var/16:.4f}")
    print(f"    Stats variance per dim: {stats_var/4:.4f}")
    print(f"    Base variance per dim: {base_var/15:.4f}")

    return {
        "avg_zero_rate": avg_zero_rate,
        "history_var_frac": history_var / total_var,
        "stats_var_frac": stats_var / total_var,
    }


# ──────────────────────────────────────────────
# Training helper with loss tracking
# ──────────────────────────────────────────────

def train_with_tracking(agent, trainer, num_sessions, batch_size=32, label=""):
    """Train and return detailed loss/eval curves."""
    print(f"\n  Training {label}: {num_sessions} sessions...")
    losses = []
    evals = []

    def callback(data):
        if data["type"] == "batch_update":
            losses.append({"episode": data["episode"], "loss": data["loss"]})
        elif data["type"] == "evaluation":
            evals.append({"episode": data["episode"], "avg_chips": data["avg_chips_per_round"]})

    start = time.time()
    trainer.train(
        num_episodes=num_sessions,
        batch_size=batch_size,
        callback=callback,
    )
    elapsed = time.time() - start

    # Final eval
    heuristic = HeuristicAgent()
    agent.set_train_mode(False)
    final_score = quick_evaluate(agent, heuristic, num_rounds=500)

    print(f"  {label} done in {elapsed:.1f}s")
    print(f"  Final eval vs heuristic (500 rounds): {final_score:+.4f}")

    return {
        "losses": losses,
        "evals": evals,
        "final_score": final_score,
        "elapsed": elapsed,
    }


# ──────────────────────────────────────────────
# Experiment 1: Isolate Network Size Effect
# ──────────────────────────────────────────────

def exp1_network_size_isolation(num_sessions=667):
    """Train AdaptiveValueAgent with wider network (128 hidden) to isolate size effect."""
    print("\n" + "=" * 60)
    print("  EXPERIMENT 1: Network Size Isolation")
    print("  AdaptiveValue with 128 hidden vs original 64 hidden")
    print("=" * 60 + "\n")

    results = {}

    # Variant A: AdaptiveValue with original 64 hidden (baseline)
    print("  --- Variant A: AdaptiveValue (19->64->64->1) [original] ---")
    agent_a = AdaptiveValueAgent()
    trainer_a = AdaptiveTrainer(agent_a, learning_rate=1e-4)
    results["adaptive_64"] = train_with_tracking(
        agent_a, trainer_a, num_sessions, label="AdaptiveValue-64"
    )

    # Variant B: AdaptiveValue with WIDER 128 hidden
    print("\n  --- Variant B: AdaptiveValue (19->128->128->1) [wider] ---")
    agent_b = AdaptiveValueAgent()
    # Override the network to use 128 hidden
    agent_b.model = ValueNetwork(agent_b.input_size, hidden_size=128)
    trainer_b = AdaptiveTrainer(agent_b, learning_rate=1e-4)
    results["adaptive_128"] = train_with_tracking(
        agent_b, trainer_b, num_sessions, label="AdaptiveValue-128"
    )

    # Variant C: AdaptiveHistory with NARROW 64 hidden (reverse test)
    print("\n  --- Variant C: AdaptiveHistory (35->64->64->1) [narrow] ---")
    agent_c = AdaptiveHistoryAgent()
    agent_c.model = ValueNetwork(agent_c.input_size, hidden_size=64)
    trainer_c = AdaptiveHistoryTrainer(agent_c, learning_rate=1e-4)
    results["history_64"] = train_with_tracking(
        agent_c, trainer_c, num_sessions, label="AdaptiveHistory-64"
    )

    # Variant D: AdaptiveHistory with original 128 hidden (reference)
    print("\n  --- Variant D: AdaptiveHistory (35->128->128->1) [original] ---")
    agent_d = AdaptiveHistoryAgent()
    trainer_d = AdaptiveHistoryTrainer(agent_d, learning_rate=1e-4)
    results["history_128"] = train_with_tracking(
        agent_d, trainer_d, num_sessions, label="AdaptiveHistory-128"
    )

    print("\n  --- EXPERIMENT 1 SUMMARY ---")
    print(f"  {'Variant':>30s}  {'Final Score':>12s}  {'Interpretation'}")
    print("  " + "-" * 80)
    for key, r in results.items():
        score = r["final_score"]
        if key == "adaptive_64":
            interp = "Baseline (should be ~+1.06)"
        elif key == "adaptive_128":
            interp = "If low: network size is the culprit"
        elif key == "history_64":
            interp = "If high: history features + narrow works"
        else:
            interp = "Reference (should be ~+0.22)"
        print(f"  {key:>30s}  {score:>+12.4f}  {interp}")

    return results


# ──────────────────────────────────────────────
# Experiment 2: Extended Training (2x duration)
# ──────────────────────────────────────────────

def exp2_extended_training(num_sessions=1334):
    """Train AdaptiveHistory for 2x longer to test if it just needs more data."""
    print("\n" + "=" * 60)
    print("  EXPERIMENT 2: Extended Training (2x duration)")
    print("  AdaptiveHistory for 1334 sessions (~40K hands)")
    print("=" * 60 + "\n")

    agent = AdaptiveHistoryAgent()
    trainer = AdaptiveHistoryTrainer(agent, learning_rate=1e-4)

    result = train_with_tracking(
        agent, trainer, num_sessions, label="AdaptiveHistory-2x"
    )

    print(f"\n  --- EXPERIMENT 2 SUMMARY ---")
    print(f"  AdaptiveHistory 2x training: {result['final_score']:+.4f}")
    print(f"  (Compare to 1x result from Exp 1)")

    return result


# ──────────────────────────────────────────────
# Experiment 5: Loss Curve / Gradient Analysis
# ──────────────────────────────────────────────

def exp5_gradient_analysis():
    """Compare gradient norms and loss dynamics for both architectures."""
    print("\n" + "=" * 60)
    print("  EXPERIMENT 5: Gradient & Loss Dynamics Analysis")
    print("=" * 60 + "\n")

    # Short training runs with gradient monitoring
    num_sessions = 100

    configs = {
        "adaptive_64": {
            "agent": AdaptiveValueAgent(),
            "hidden": 64,
        },
        "adaptive_128": {
            "agent": AdaptiveValueAgent(),
            "hidden": 128,
        },
        "history_128": {
            "agent": AdaptiveHistoryAgent(),
            "hidden": 128,
        },
    }

    results = {}

    for name, cfg in configs.items():
        agent = cfg["agent"]
        if cfg["hidden"] != 64 and isinstance(agent, AdaptiveValueAgent) and not isinstance(agent, AdaptiveHistoryAgent):
            agent.model = ValueNetwork(agent.input_size, hidden_size=128)

        if isinstance(agent, AdaptiveHistoryAgent):
            trainer = AdaptiveHistoryTrainer(agent, learning_rate=1e-4)
        else:
            trainer = AdaptiveTrainer(agent, learning_rate=1e-4)

        agent.set_train_mode(True)

        grad_norms = []
        losses = []
        weight_norms = []
        output_scales = []

        for session_idx in range(num_sessions):
            session_data = trainer.collect_episode()

            # Sample output scale from the batch
            with torch.no_grad():
                for chains, rewards in session_data[:3]:
                    for p_idx in [0, 1]:
                        for enc in chains[p_idx][:2]:
                            val = agent.model(enc).item()
                            output_scales.append(abs(val))

            batch_data = session_data
            if len(batch_data) >= 32:
                # Manual update to capture gradients
                trainer.optimizer.zero_grad()
                total_losses = []

                for chains, rewards in batch_data:
                    for p_idx in [0, 1]:
                        chain = chains[p_idx]
                        if not chain:
                            continue
                        for t in range(len(chain)):
                            prediction = agent.model(chain[t]).squeeze(0)
                            if t == len(chain) - 1:
                                target = torch.FloatTensor([rewards[p_idx]])
                            else:
                                with torch.no_grad():
                                    target = agent.model(chain[t + 1]).squeeze(0)
                            loss = torch.nn.MSELoss()(prediction, target)
                            total_losses.append(loss)

                if total_losses:
                    mean_loss = torch.stack(total_losses).mean()
                    mean_loss.backward()

                    # Capture gradient norms per layer
                    session_grad_norms = {}
                    total_grad_norm = 0
                    for pname, p in agent.model.named_parameters():
                        if p.grad is not None:
                            gn = p.grad.norm().item()
                            session_grad_norms[pname] = gn
                            total_grad_norm += gn ** 2
                    grad_norms.append({
                        "session": session_idx,
                        "total_grad_norm": math.sqrt(total_grad_norm),
                        "per_layer": session_grad_norms,
                        "loss": mean_loss.item(),
                    })
                    losses.append(mean_loss.item())

                    trainer.optimizer.step()

                # Weight norms
                wn = {}
                for pname, p in agent.model.named_parameters():
                    wn[pname] = p.data.norm().item()
                weight_norms.append(wn)

                batch_data = []

        results[name] = {
            "grad_norms": grad_norms,
            "losses": losses,
            "weight_norms": weight_norms,
            "output_scales": output_scales,
        }

        # Print summary
        if grad_norms:
            early_grads = [g["total_grad_norm"] for g in grad_norms[:10]]
            late_grads = [g["total_grad_norm"] for g in grad_norms[-10:]]
            early_loss = [g["loss"] for g in grad_norms[:10]]
            late_loss = [g["loss"] for g in grad_norms[-10:]]

            print(f"\n  {name}:")
            print(f"    Grad norm  - early: {np.mean(early_grads):.4f}, late: {np.mean(late_grads):.4f}")
            print(f"    Loss       - early: {np.mean(early_loss):.4f}, late: {np.mean(late_loss):.4f}")
            if output_scales:
                print(f"    Output scale - mean: {np.mean(output_scales):.4f}, max: {np.max(output_scales):.4f}")

            # Per-layer gradient norms
            print(f"    Per-layer grad norms (last update):")
            last_grads = grad_norms[-1]["per_layer"]
            for lname, gn in last_grads.items():
                print(f"      {lname}: {gn:.6f}")

    return results


# ──────────────────────────────────────────────
# Experiment 6: Initial Value Scale Check
# ──────────────────────────────────────────────

def exp6_initial_value_scale():
    """Check if random initialization produces different value scales for different architectures."""
    print("\n" + "=" * 60)
    print("  EXPERIMENT 6: Initial Value Scale (Random Initialization)")
    print("=" * 60 + "\n")

    session = PokerSession()
    n_samples = 200

    configs = {
        "AdaptiveValue-64": (AdaptiveValueAgent, 64),
        "AdaptiveValue-128": (AdaptiveValueAgent, 128),
        "AdaptiveHistory-64": (AdaptiveHistoryAgent, 64),
        "AdaptiveHistory-128": (AdaptiveHistoryAgent, 128),
    }

    for name, (agent_cls, hidden) in configs.items():
        values = []
        spreads = []  # value spread per decision (max - min across legal actions)

        for trial in range(5):  # 5 random initializations
            agent = agent_cls()
            if hidden != (64 if isinstance(agent, AdaptiveValueAgent) and not isinstance(agent, AdaptiveHistoryAgent) else 128):
                agent.model = ValueNetwork(agent.input_size, hidden_size=hidden)

            session.reset()
            for _ in range(n_samples // 5):
                session.new_hand()
                while not session.is_finished:
                    player = session.current_player
                    obs = session.get_observation(viewer_id=player)
                    evals = agent.get_action_evaluations(obs)
                    vals = [e["value"] for e in evals]
                    values.extend(vals)
                    if len(vals) > 1:
                        spreads.append(max(vals) - min(vals))
                    action = agent.select_action(obs)
                    session.step(action)

        values = np.array(values)
        spreads = np.array(spreads) if spreads else np.array([0])
        print(f"  {name}:")
        print(f"    Value range: [{values.min():.4f}, {values.max():.4f}]")
        print(f"    Value mean: {values.mean():.4f}, std: {values.std():.4f}")
        print(f"    Action spread (max-min per decision): mean={spreads.mean():.4f}, std={spreads.std():.4f}")
        print()


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  DIAGNOSTIC: Why AdaptiveHistory Underperforms AdaptiveValue")
    print("=" * 60)

    # Exp 3: Parameter counts (instant, run first)
    param_results = exp3_parameter_counts()

    # Exp 6: Initial value scale (fast)
    exp6_initial_value_scale()

    # Exp 4: History feature analysis (fast)
    feature_results = exp4_history_feature_analysis(num_sessions=50)

    # Exp 5: Gradient analysis (moderate)
    gradient_results = exp5_gradient_analysis()

    # Exp 1: Network size isolation (main experiment, 4 training runs)
    exp1_results = exp1_network_size_isolation(num_sessions=667)

    # Exp 2: Extended training (1 long training run)
    exp2_result = exp2_extended_training(num_sessions=1334)

    # ──────────────────────────────────────────────
    # Final Summary
    # ──────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  FINAL DIAGNOSTIC SUMMARY")
    print("=" * 60 + "\n")

    print("  Experiment 1 — Network Size Isolation:")
    for key, r in exp1_results.items():
        print(f"    {key:>30s}: {r['final_score']:+.4f}")

    print(f"\n  Experiment 2 — Extended Training:")
    print(f"    AdaptiveHistory 2x: {exp2_result['final_score']:+.4f}")

    print(f"\n  Experiment 4 — History Feature Informativeness:")
    print(f"    Avg zero rate: {feature_results['avg_zero_rate']:.1f}%")
    print(f"    History variance fraction: {feature_results['history_var_frac']*100:.1f}%")
    print(f"    Stats variance fraction: {feature_results['stats_var_frac']*100:.1f}%")

    # Diagnosis
    print("\n  DIAGNOSIS:")
    a64 = exp1_results["adaptive_64"]["final_score"]
    a128 = exp1_results["adaptive_128"]["final_score"]
    h64 = exp1_results["history_64"]["final_score"]
    h128 = exp1_results["history_128"]["final_score"]
    h2x = exp2_result["final_score"]

    if a128 < a64 - 0.3:
        print("  -> H1 CONFIRMED: Wider network alone causes significant degradation.")
        print(f"     AdaptiveValue-64: {a64:+.4f} vs AdaptiveValue-128: {a128:+.4f}")
    else:
        print("  -> H1 REJECTED: Wider network alone does NOT cause major degradation.")
        print(f"     AdaptiveValue-64: {a64:+.4f} vs AdaptiveValue-128: {a128:+.4f}")

    if h2x > h128 + 0.3:
        print("  -> H2 (data): Extended training helps significantly — model was underfitting.")
        print(f"     History-128 1x: {h128:+.4f} vs History-128 2x: {h2x:+.4f}")
    else:
        print("  -> H2 (data): Extended training does NOT help much — not a simple underfitting issue.")
        print(f"     History-128 1x: {h128:+.4f} vs History-128 2x: {h2x:+.4f}")

    if feature_results["avg_zero_rate"] > 60:
        print(f"  -> H2 (noise): History features are {feature_results['avg_zero_rate']:.0f}% zeros — likely noise.")
    else:
        print(f"  -> H2 (noise): History features are NOT mostly zeros ({feature_results['avg_zero_rate']:.0f}% zero).")

    if h64 > h128 + 0.2:
        print("  -> INTERACTION: History + narrow outperforms history + wide — capacity harm is real.")
    elif h64 < h128:
        print("  -> INTERACTION: Wide network doesn't hurt history features — something else is going on.")

    print("\n  Done.")


if __name__ == "__main__":
    main()
