"""
Diagnostic experiment: Why does PopAdaptiveAgent (-0.64 avg) underperform
its parent AdaptiveValueAgent (+1.06 avg)?

Hypotheses:
  H1: Opponent pool is too weak — teaches exploitation of weak opponents
  H2: Rotation disrupts stat accumulation
  H3: Agent learns mediocre "average" strategy
  H4: Model files don't load correctly
  H5: Self-play's moving target actually works well
  H6: Half training signal — only player 0 chains recorded (vs both in self-play)

Experiments:
  1. Verify opponent models load and play distinctly
  2. Track stat diversity during training against different opponents
  3. Train pop_adaptive with ONLY strong opponents (adaptive_value + self-snapshots)
  4. Train regular adaptive_value as baseline with same budget (667 sessions)
  5. Examine rotation/snapshot arithmetic
  6. Measure training data volume: pop_adaptive vs adaptive_value per session
  7. Train pop_adaptive with both-player chains (fix H6)
"""

import os
import sys
import copy
import time
import json
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.agents.heuristic import HeuristicAgent
from src.agents.value_based import ValueBasedAgent
from src.agents.adaptive_value import AdaptiveValueAgent
from src.agents.pop_adaptive import PopAdaptiveAgent
from src.engine.poker_session import PokerSession
from src.engine.leduc_game import LeducGame, Action
from src.training.adaptive_trainer import AdaptiveTrainer
from src.training.pop_adaptive_trainer import PopAdaptiveTrainer
from src.training.evaluation import evaluate_agents, quick_evaluate, compute_robustness_metrics
from dataclasses import replace


def separator(title):
    print(f"\n{'='*70}")
    print(f"  EXPERIMENT: {title}")
    print(f"{'='*70}\n")


# ──────────────────────────────────────────────────────────────────────
# Experiment 1: Verify opponent models load and play distinctly
# ──────────────────────────────────────────────────────────────────────
def exp1_verify_opponent_loading():
    separator("1 — Verify opponent models load and play distinctly")

    # Check model file existence
    vb_path = "models/value_based_agent.pt"
    av_path = "models/adaptive_value_agent.pt"
    print(f"  value_based model exists:    {os.path.exists(vb_path)}")
    print(f"  adaptive_value model exists:  {os.path.exists(av_path)}")

    # Load opponents same way as PopAdaptiveTrainer
    heuristic = HeuristicAgent()

    vb = ValueBasedAgent()
    if os.path.exists(vb_path):
        vb.load_model(vb_path)
        print("  value_based model loaded successfully")
    vb.set_train_mode(False)

    av = AdaptiveValueAgent()
    if os.path.exists(av_path):
        av.load_model(av_path)
        print("  adaptive_value model loaded successfully")
    av.set_train_mode(False)

    # Play 200 hands with each opponent and check their action distributions
    opponents = {"heuristic": heuristic, "value_based": vb, "adaptive_value": av}
    action_profiles = {}

    for name, opp in opponents.items():
        session = PokerSession()
        action_counts = {"FOLD": 0, "CALL": 0, "RAISE": 0}
        total_actions = 0

        for _ in range(200):
            session.new_hand()
            while not session.is_finished:
                player = session.current_player
                obs = session.get_observation(viewer_id=player)
                if player == 1:  # opponent
                    action = opp.select_action(obs)
                    if isinstance(action, tuple):
                        action = action[0]
                    action_counts[action.name] += 1
                    total_actions += 1
                else:  # random player 0 for testing
                    action = heuristic.select_action(obs)
                    if isinstance(action, tuple):
                        action = action[0]
                session.step(action)

        profile = {a: round(c / total_actions, 3) for a, c in action_counts.items()}
        action_profiles[name] = profile
        print(f"\n  {name:20s} action profile (200 hands vs heuristic):")
        print(f"    FOLD={profile['FOLD']:.3f}  CALL={profile['CALL']:.3f}  RAISE={profile['RAISE']:.3f}")

    # Check if profiles are actually different
    fold_rates = [p["FOLD"] for p in action_profiles.values()]
    if max(fold_rates) - min(fold_rates) > 0.05:
        print("\n  RESULT: Opponents have meaningfully different play styles")
    else:
        print("\n  WARNING: Opponents play very similarly — pool lacks diversity!")

    return action_profiles


# ──────────────────────────────────────────────────────────────────────
# Experiment 2: Track stat diversity during pop_adaptive training
# ──────────────────────────────────────────────────────────────────────
def exp2_stat_diversity():
    separator("2 — Track stat diversity during training")

    agent = PopAdaptiveAgent()
    trainer = PopAdaptiveTrainer(agent, learning_rate=1e-4, hands_per_session=30,
                                  rotate_every=100, snapshot_every=500)

    stat_log = []

    # Run 30 sessions (=900 hands) and log opponent stats at end of each
    for sess_idx in range(30):
        trainer.session.reset()
        opponent = trainer._get_current_opponent()
        opp_name = trainer._get_current_opponent_name()

        for _ in range(30):
            trainer.session.new_hand()
            while not trainer.session.is_finished:
                player = trainer.session.current_player
                obs = trainer.session.get_observation(viewer_id=player)
                if player == 0:
                    action = agent.select_action(obs)
                else:
                    action = opponent.select_action(obs)
                if isinstance(action, tuple):
                    action = action[0]
                trainer.session.step(action)

        # Log the stats player 0 accumulated about the opponent
        stats_p0 = trainer.session.stats[0].to_feature_vector()
        stat_log.append({
            "session": sess_idx,
            "opponent": opp_name,
            "episode_count": trainer.episode_count,
            "stats_player0_sees": stats_p0,
        })
        trainer.episode_count += 30
        trainer._maybe_rotate_opponent()
        trainer._maybe_snapshot_self()

    print(f"  Logged {len(stat_log)} sessions\n")
    print(f"  {'Sess':>4s}  {'Opponent':>20s}  {'Ep#':>5s}  fold_r  raise_r  f2r_r  conf")
    print(f"  {'-'*75}")
    for entry in stat_log:
        s = entry["stats_player0_sees"]
        print(f"  {entry['session']:>4d}  {entry['opponent']:>20s}  {entry['episode_count']:>5d}  "
              f"{s[0]:.3f}   {s[1]:.3f}    {s[2]:.3f}  {s[3]:.3f}")

    # Summarize stat ranges by opponent type
    print(f"\n  Per-opponent stat ranges:")
    from collections import defaultdict
    by_opp = defaultdict(list)
    for entry in stat_log:
        by_opp[entry["opponent"]].append(entry["stats_player0_sees"])

    for opp_name, stats_list in by_opp.items():
        fold_rates = [s[0] for s in stats_list]
        raise_rates = [s[1] for s in stats_list]
        print(f"    {opp_name:>20s}: fold_rate [{min(fold_rates):.3f} - {max(fold_rates):.3f}], "
              f"raise_rate [{min(raise_rates):.3f} - {max(raise_rates):.3f}]")

    return stat_log


# ──────────────────────────────────────────────────────────────────────
# Experiment 3: Rotation/snapshot arithmetic
# ──────────────────────────────────────────────────────────────────────
def exp3_rotation_arithmetic():
    separator("3 — Rotation and snapshot arithmetic")

    hands_per_session = 30
    total_sessions = 667
    rotate_every = 100  # in episodes (= hands)
    snapshot_every = 500

    total_hands = total_sessions * hands_per_session
    rotations = total_hands // rotate_every
    snapshots = total_hands // snapshot_every

    sessions_per_rotation = rotate_every / hands_per_session

    print(f"  Total sessions:      {total_sessions}")
    print(f"  Total hands:         {total_hands}")
    print(f"  Rotations:           {rotations} (every {rotate_every} hands = {sessions_per_rotation:.1f} sessions)")
    print(f"  Snapshots added:     {snapshots}")
    print(f"  Initial pool size:   3 (heuristic, value_based, adaptive_value)")
    print(f"  Final pool size:     {3 + snapshots}")
    print()

    # Simulate the rotation schedule
    episode_count = 0
    opponent_idx = 0
    pool_size = 3
    opp_names = ["heuristic", "value_based", "adaptive_value"]
    sessions_per_opp = {}

    for sess in range(total_sessions):
        current_name = opp_names[opponent_idx % len(opp_names)] if opponent_idx < len(opp_names) else f"self_snapshot"
        sessions_per_opp[current_name] = sessions_per_opp.get(current_name, 0) + 1

        episode_count += hands_per_session
        if episode_count % rotate_every == 0:
            opponent_idx = (opponent_idx + 1) % pool_size
        if episode_count % snapshot_every == 0:
            pool_size += 1
            opp_names.append(f"self_snapshot_{episode_count}")

    print(f"  Sessions per opponent type:")
    total_snap_sessions = 0
    for name, count in sorted(sessions_per_opp.items()):
        pct = 100 * count / total_sessions
        print(f"    {name:>30s}: {count:>4d} sessions ({pct:.1f}%)")
        if "snapshot" in name:
            total_snap_sessions += count

    print(f"\n  Total self-snapshot sessions: {total_snap_sessions}")
    print(f"  Total non-snapshot sessions:  {total_sessions - total_snap_sessions}")


# ──────────────────────────────────────────────────────────────────────
# Experiment 4: CRITICAL — Training data volume comparison
# ──────────────────────────────────────────────────────────────────────
def exp4_training_data_volume():
    separator("4 — Training data volume: pop_adaptive vs adaptive_value")

    # AdaptiveTrainer: self-play, BOTH players' chains recorded
    agent_av = AdaptiveValueAgent()
    trainer_av = AdaptiveTrainer(agent_av, learning_rate=1e-4, hands_per_session=30)
    agent_av.set_train_mode(True)

    # Collect one session
    session_data_av = trainer_av.collect_episode()
    total_chains_av = sum(len(chains[0]) + len(chains[1]) for chains, rewards in session_data_av)
    p0_chains_av = sum(len(chains[0]) for chains, rewards in session_data_av)
    p1_chains_av = sum(len(chains[1]) for chains, rewards in session_data_av)

    print(f"  AdaptiveTrainer (self-play) — 1 session of 30 hands:")
    print(f"    Player 0 chain entries: {p0_chains_av}")
    print(f"    Player 1 chain entries: {p1_chains_av}")
    print(f"    Total chain entries:    {total_chains_av}")

    # PopAdaptiveTrainer: only player 0's chain recorded
    agent_pop = PopAdaptiveAgent()
    trainer_pop = PopAdaptiveTrainer(agent_pop, learning_rate=1e-4, hands_per_session=30)
    agent_pop.set_train_mode(True)

    session_data_pop = trainer_pop.collect_episode()
    total_chains_pop = sum(len(chains[0]) + len(chains[1]) for chains, rewards in session_data_pop)
    p0_chains_pop = sum(len(chains[0]) for chains, rewards in session_data_pop)
    p1_chains_pop = sum(len(chains[1]) for chains, rewards in session_data_pop)

    print(f"\n  PopAdaptiveTrainer (pool) — 1 session of 30 hands:")
    print(f"    Player 0 chain entries: {p0_chains_pop}")
    print(f"    Player 1 chain entries: {p1_chains_pop}")
    print(f"    Total chain entries:    {total_chains_pop}")

    ratio = total_chains_pop / total_chains_av if total_chains_av > 0 else 0
    print(f"\n  Data ratio (pop / adaptive): {ratio:.2f}x")
    if ratio < 0.6:
        print("  >>> CRITICAL FINDING: PopAdaptive gets HALF the training data!")
        print("  >>> In self-play, BOTH players use the same network, so both chains train it.")
        print("  >>> In pool training, only player 0 (the learner) generates training signal.")

    return {
        "adaptive_total": total_chains_av,
        "pop_total": total_chains_pop,
        "ratio": ratio,
    }


# ──────────────────────────────────────────────────────────────────────
# Experiment 5: Train adaptive_value baseline (667 sessions, same budget)
# ──────────────────────────────────────────────────────────────────────
def exp5_train_adaptive_value_baseline():
    separator("5 — Train fresh AdaptiveValue (667 sessions baseline)")

    agent = AdaptiveValueAgent()
    trainer = AdaptiveTrainer(agent, learning_rate=1e-4, hands_per_session=30)

    losses = []
    eval_scores = []

    def callback(data):
        if data["type"] == "batch_update":
            losses.append(data["loss"])
        elif data["type"] == "evaluation":
            eval_scores.append(data["avg_chips_per_round"])

    start = time.time()
    trainer.train(num_episodes=667, batch_size=32,
                  save_path="models/diag_adaptive_value_fresh.pt",
                  callback=callback)
    elapsed = time.time() - start

    print(f"\n  Training time: {elapsed:.1f}s")
    print(f"  Final loss: {losses[-1]:.4f}" if losses else "  No losses recorded")
    if eval_scores:
        print(f"  Final eval vs heuristic: {eval_scores[-1]:+.3f}")
        print(f"  Best eval vs heuristic:  {max(eval_scores):+.3f}")

    return {"losses": losses, "evals": eval_scores, "time": elapsed}


# ──────────────────────────────────────────────────────────────────────
# Experiment 6: Train pop_adaptive from scratch (reproduction)
# ──────────────────────────────────────────────────────────────────────
def exp6_train_pop_adaptive_fresh():
    separator("6 — Train fresh PopAdaptive (667 sessions)")

    agent = PopAdaptiveAgent()
    trainer = PopAdaptiveTrainer(agent, learning_rate=1e-4, hands_per_session=30)

    losses = []
    eval_scores = []

    def callback(data):
        if data["type"] == "batch_update":
            losses.append(data["loss"])
        elif data["type"] == "evaluation":
            eval_scores.append(data["avg_chips_per_round"])

    start = time.time()
    trainer.train(num_episodes=667, batch_size=32,
                  save_path="models/diag_pop_adaptive_fresh.pt",
                  callback=callback)
    elapsed = time.time() - start

    print(f"\n  Training time: {elapsed:.1f}s")
    print(f"  Final loss: {losses[-1]:.4f}" if losses else "  No losses recorded")
    if eval_scores:
        print(f"  Final eval vs heuristic: {eval_scores[-1]:+.3f}")
        print(f"  Best eval vs heuristic:  {max(eval_scores):+.3f}")

    return {"losses": losses, "evals": eval_scores, "time": elapsed}


# ──────────────────────────────────────────────────────────────────────
# Experiment 7: Train pop_adaptive with ONLY strong opponents
# ──────────────────────────────────────────────────────────────────────
def exp7_strong_opponents_only():
    separator("7 — Train PopAdaptive with ONLY strong opponents")

    agent = PopAdaptiveAgent()
    trainer = PopAdaptiveTrainer(agent, learning_rate=1e-4, hands_per_session=30)

    # Override pool: remove heuristic and value_based, keep only adaptive_value
    av = AdaptiveValueAgent()
    av_path = "models/adaptive_value_agent.pt"
    if os.path.exists(av_path):
        av.load_model(av_path)
    av.set_train_mode(False)
    trainer.opponent_pool = [("adaptive_value", av)]

    losses = []
    eval_scores = []

    def callback(data):
        if data["type"] == "batch_update":
            losses.append(data["loss"])
        elif data["type"] == "evaluation":
            eval_scores.append(data["avg_chips_per_round"])

    start = time.time()
    trainer.train(num_episodes=667, batch_size=32,
                  save_path="models/diag_pop_strong_only.pt",
                  callback=callback)
    elapsed = time.time() - start

    print(f"\n  Training time: {elapsed:.1f}s")
    print(f"  Final loss: {losses[-1]:.4f}" if losses else "  No losses recorded")
    if eval_scores:
        print(f"  Final eval vs heuristic: {eval_scores[-1]:+.3f}")
        print(f"  Best eval vs heuristic:  {max(eval_scores):+.3f}")

    return {"losses": losses, "evals": eval_scores, "time": elapsed}


# ──────────────────────────────────────────────────────────────────────
# Experiment 8: Head-to-head comparison of all trained variants
# ──────────────────────────────────────────────────────────────────────
def exp8_head_to_head():
    separator("8 — Head-to-head comparison of all variants")

    agents = {}

    # Load existing pretrained
    av_pretrained = AdaptiveValueAgent()
    av_pretrained.load_model("models/adaptive_value_agent.pt")
    av_pretrained.set_train_mode(False)
    agents["adaptive_value_pretrained"] = av_pretrained

    pop_pretrained = PopAdaptiveAgent()
    pop_pretrained.load_model("models/pop_adaptive_agent.pt")
    pop_pretrained.set_train_mode(False)
    agents["pop_adaptive_pretrained"] = pop_pretrained

    # Load freshly trained
    for name, path in [
        ("adaptive_value_fresh", "models/diag_adaptive_value_fresh.pt"),
        ("pop_adaptive_fresh", "models/diag_pop_adaptive_fresh.pt"),
        ("pop_strong_only", "models/diag_pop_strong_only.pt"),
    ]:
        if os.path.exists(path):
            a = AdaptiveValueAgent()  # same architecture
            a.load_model(path)
            a.set_train_mode(False)
            agents[name] = a

    # Also include baselines
    agents["heuristic"] = HeuristicAgent()

    vb = ValueBasedAgent()
    if os.path.exists("models/value_based_agent.pt"):
        vb.load_model("models/value_based_agent.pt")
    vb.set_train_mode(False)
    agents["value_based"] = vb

    # Round-robin
    agent_names = list(agents.keys())
    results = {a: {} for a in agent_names}

    for i, a0_name in enumerate(agent_names):
        results[a0_name][a0_name] = 0.0
        for j, a1_name in enumerate(agent_names):
            if j <= i:
                continue
            result = evaluate_agents(agents[a0_name], agents[a1_name], num_rounds=500)
            results[a0_name][a1_name] = round(result.agent_0_avg_chips, 4)
            results[a1_name][a0_name] = round(result.agent_1_avg_chips, 4)

    # Print results
    print(f"\n  {'Agent':>30s} |", end="")
    for name in agent_names:
        print(f" {name[:12]:>12s}", end="")
    print(" |     AVG")
    print(f"  {'-'*30}-+" + "-" * (13 * len(agent_names)) + "-+--------")

    for a0 in agent_names:
        print(f"  {a0:>30s} |", end="")
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

    # Print robustness rankings
    print(f"\n  Robustness rankings:")
    metrics = {}
    for name in agent_names:
        opp_scores = {o: results[name].get(o, 0.0) for o in agent_names if o != name}
        metrics[name] = compute_robustness_metrics(opp_scores)

    sorted_agents = sorted(agent_names, key=lambda a: (metrics[a]["avg"],), reverse=True)
    print(f"  {'Rank':>4s}  {'Agent':>30s}  {'Avg':>8s}  {'Worst':>8s}  {'Std':>8s}")
    print(f"  {'-'*70}")
    for rank, name in enumerate(sorted_agents, 1):
        m = metrics[name]
        print(f"  {rank:>4d}  {name:>30s}  {m['avg']:+8.4f}  {m['worst_case']:+8.4f}  {m['std']:8.4f}")

    return results


# ──────────────────────────────────────────────────────────────────────
# Experiment 9: Measure loss convergence difference
# ──────────────────────────────────────────────────────────────────────
def exp9_loss_trajectory_comparison():
    separator("9 — Loss trajectory comparison (adaptive vs pop)")

    # Train adaptive_value for 200 sessions, recording every batch loss
    agent_av = AdaptiveValueAgent()
    trainer_av = AdaptiveTrainer(agent_av, learning_rate=1e-4, hands_per_session=30)
    losses_av = []

    def cb_av(data):
        if data["type"] == "batch_update":
            losses_av.append(data["loss"])

    agent_av.set_train_mode(True)
    # Manual training to control exactly
    batch_data = []
    for sess in range(200):
        session_data = trainer_av.collect_episode()
        batch_data.extend(session_data)
        if len(batch_data) >= 32:
            loss = trainer_av.update_model(batch_data)
            losses_av.append(loss)
            batch_data = []

    # Train pop_adaptive for 200 sessions
    agent_pop = PopAdaptiveAgent()
    trainer_pop = PopAdaptiveTrainer(agent_pop, learning_rate=1e-4, hands_per_session=30)
    losses_pop = []

    agent_pop.set_train_mode(True)
    batch_data = []
    for sess in range(200):
        session_data = trainer_pop.collect_episode()
        batch_data.extend(session_data)
        if len(batch_data) >= 32:
            loss = trainer_pop.update_model(batch_data)
            losses_pop.append(loss)
            batch_data = []

    # Compare loss trajectories
    def avg_window(lst, window=5):
        return [sum(lst[i:i+window])/window for i in range(0, len(lst)-window+1, window)]

    av_smoothed = avg_window(losses_av)
    pop_smoothed = avg_window(losses_pop)

    print(f"  AdaptiveValue loss trajectory (windowed avg of 5):")
    for i, l in enumerate(av_smoothed[:15]):
        print(f"    batch {i*5:>3d}-{i*5+4:>3d}: {l:.4f}")

    print(f"\n  PopAdaptive loss trajectory (windowed avg of 5):")
    for i, l in enumerate(pop_smoothed[:15]):
        print(f"    batch {i*5:>3d}-{i*5+4:>3d}: {l:.4f}")

    if av_smoothed and pop_smoothed:
        av_final = sum(av_smoothed[-3:]) / 3 if len(av_smoothed) >= 3 else av_smoothed[-1]
        pop_final = sum(pop_smoothed[-3:]) / 3 if len(pop_smoothed) >= 3 else pop_smoothed[-1]
        print(f"\n  Final avg loss — AdaptiveValue: {av_final:.4f}, PopAdaptive: {pop_final:.4f}")

    return {"av_losses": losses_av, "pop_losses": losses_pop}


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────
def main():
    print("=" * 70)
    print("  DIAGNOSIS: Why does PopAdaptive underperform AdaptiveValue?")
    print("=" * 70)

    all_results = {}

    # Quick diagnostics first
    all_results["exp1_loading"] = exp1_verify_opponent_loading()
    all_results["exp2_stats"] = exp2_stat_diversity()
    all_results["exp3_arithmetic"] = exp3_rotation_arithmetic()
    all_results["exp4_data_volume"] = exp4_training_data_volume()

    # Loss trajectory comparison
    all_results["exp9_loss"] = exp9_loss_trajectory_comparison()

    # Training experiments (slower)
    all_results["exp5_av_baseline"] = exp5_train_adaptive_value_baseline()
    all_results["exp6_pop_fresh"] = exp6_train_pop_adaptive_fresh()
    all_results["exp7_strong_only"] = exp7_strong_opponents_only()

    # Head-to-head evaluation
    all_results["exp8_h2h"] = exp8_head_to_head()

    # Save results
    results_path = "experiments/diagnose_pop_adaptive_results.json"

    # Convert non-serializable data
    serializable = {}
    for k, v in all_results.items():
        if k == "exp4_data_volume":
            serializable[k] = v
        elif k in ("exp5_av_baseline", "exp6_pop_fresh", "exp7_strong_only"):
            serializable[k] = {
                "final_loss": v["losses"][-1] if v["losses"] else None,
                "n_losses": len(v["losses"]),
                "evals": v["evals"],
                "time": v["time"],
            }
        elif k == "exp8_h2h":
            serializable[k] = v
        elif k == "exp9_loss":
            serializable[k] = {
                "av_n_batches": len(v["av_losses"]),
                "pop_n_batches": len(v["pop_losses"]),
                "av_final_loss": v["av_losses"][-1] if v["av_losses"] else None,
                "pop_final_loss": v["pop_losses"][-1] if v["pop_losses"] else None,
            }

    with open(results_path, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"\nResults saved to {results_path}")

    # ── Summary ──
    separator("SUMMARY OF FINDINGS")

    dv = all_results["exp4_data_volume"]
    print(f"  1. Training data ratio (pop/adaptive): {dv['ratio']:.2f}x")
    if dv["ratio"] < 0.6:
        print("     >>> PopAdaptive gets ~HALF the training signal per session")
        print("     >>> Self-play: both P0 and P1 use the same net, both chains train it")
        print("     >>> Pool play: only P0's chain trains the net, P1 is a frozen opponent")

    print()
    print("  2. Rotation arithmetic:")
    print("     Each opponent is seen for only ~3.3 sessions before rotation")
    print("     Stats reset per session, so no disruption from rotation itself")
    print("     But: the agent sees a chaotic mix of very different opponent styles")

    print()
    print("  3. Loss/convergence comparison (check output above)")

    print()
    print("  4. Head-to-head results (check output above)")


if __name__ == "__main__":
    main()
